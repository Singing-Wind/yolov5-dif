import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer,GPT2Config

from scipy.optimize import linear_sum_assignment


class CLIPDetectionModel(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", gpt2_model_name="gpt2",
                 block_size=1, k=80, num_classes=20,np_max=200,thresh=0.05):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=False)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        #
        # 自定义小模型配置
        # config = GPT2Config(
        #     vocab_size=50257,       # 保持不变
        #     n_positions=256,        # 缩小序列长度
        #     n_embd=512,             # embedding 和 hidden size
        #     n_layer=4,              # 层数减少
        #     n_head=4,               # 注意力头减少
        # )
        # # 初始化模型（未预训练）
        # self.gpt2 = GPT2LMHeadModel(config) #"distilgpt2"
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        
        self.block_size = block_size
        self.k = k
        self.num_bins = k + 1
        self.Np = block_size * block_size
        self.clip_feature_size = 512

        # 映射 clip image_feature -> gpt embedding size
        self.clip_proj = nn.Linear(self.clip_feature_size, self.gpt2.config.n_embd)

        # 输出 head: 1 + 4(k+1) + nc
        self.set_nc(num_classes)

        self.np_max = np_max

        self.thresh = thresh

        # 位置向量 [0, 1/k, 2/k, ..., 1]
        self.register_buffer("pos_bins", torch.linspace(0, 1, steps=self.num_bins).float())
    
    def to(self, device):
        # 重写to方法确保所有组件都被正确转移
        self = super().to(device)
        self.clip = self.clip.to(device)
        self.gpt2 = self.gpt2.to(device)
        self.clip_proj = self.clip_proj.to(device)
        self.head = self.head.to(device)
        # processor保持在CPU上
        # self.processor = self.processor.to('cpu')
        return self
    
    def set_nc(self,num_classes,device=None):
        self.nc = num_classes #输出 head: (k+1)*4 + nc
        self.head = nn.Linear(self.gpt2.config.n_embd, 1 + self.num_bins * 4 + self.nc) #1(obj)+4(k+1)(num_bins) + nc
        if device is not None:
            self.head.to(device)

    def space_to_depth(self, x, block_size):
        # [B, C, H, W] -> [B, Np, C, H//bs, W//bs]
        B, C, H, W = x.shape
        assert H % block_size == 0 and W % block_size == 0
        x = x.view(B, C, H // block_size, block_size, W // block_size, block_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(B, block_size * block_size, C, H // block_size, W // block_size)
        return x

    def forward(self, images, target=None):
        # images: [B, 3, H, W]
        # target: [bnt,10=1(b)+1(cls)+8(xyxyxyxy)] for obb
        #       : [bnt,6=1(b)+1(cls)+4(xywh)] for hbb
        #     - 'boxes': [B, T, 4]  in [x,y,w,h] normalized [0,1]
        #     - 'labels': [B, T]    in 0 ~ (nc-1)
        device = images.device
        B = images.size(0)

        # 1. Patch 处理
        if self.block_size > 1:
            patches = self.space_to_depth(images, self.block_size) #patches[B,Np,C=3,Hs,Ws]
            B, Np, C, H_patch, W_patch = patches.shape
            patches = patches.view(B * Np, C, H_patch, W_patch)#patches[B*Np,C=3,Hs,Ws]
        else:
            patches = images
            Np = 1

        # 2. Clip 特征
        inputs = self.processor(images=patches, return_tensors="pt", padding=True, do_rescale=False).to(device)
        #inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.clip.get_image_features(**inputs)  # [B*Np, 512]
        image_features = image_features.view(B, Np, self.clip_feature_size)# [B,Np, 512]
        prefix_embed = self.clip_proj(image_features)  # [B, Np, nembd=768]

        # 3. Prepare GPT input
        dummy_input = torch.ones(B, 1).long().to(device) * self.tokenizer.pad_token_id #dummy_input[B,1]
        gpt_embeds = self.gpt2.transformer.wte(dummy_input)  # gpt_embeds[B, 1, nembd=768]
        decoder_input = gpt_embeds.repeat(1, self.np_max, 1)  # [B, np_max, nembd=768]

        # 4. Concatenate prefix
        inputs_embeds = torch.cat([prefix_embed, decoder_input], dim=1)  # [B, Np + np_max, nembd=768]
        hidden = self.gpt2.transformer(inputs_embeds=inputs_embeds).last_hidden_state  # [B, Np + np_max, nembd=768]
        hidden = hidden[:, Np:, :]  # Keep only target part [B, np_max, nembd=768]

        # 5. Output
        out = self.head(hidden)  # out[B, np_max, 1 + 4(k+1) + nc]
        #obj
        objs = out[..., 0 : 1] # objs[B, np_max, 1]
        #box
        bins = out[..., 1 : 1 + 4*self.num_bins].view(B, -1, 4, self.num_bins) #bins[B, np_max, 4, k+1]
        prob_bins = F.softmax(bins, dim=-1)  #prob_bins[B, np_max, 4, k+1]
        pred_boxes = torch.sum(prob_bins * self.pos_bins, dim=-1) # prob_bins[B,np_max,4,k+1]*pos_bins[k+1] -> pred_boxes[B, np_max, 4]
        #cls
        cls_logits = out[..., 1 + 4*self.num_bins:]  # cls_logits[B, np_max, nc]
        assert cls_logits.shape[-1] == self.nc
        pred_clss = F.log_softmax(cls_logits, dim=-1)# pred_clss[B, np_max, nc]

        if target is not None:
            preds = objs,pred_boxes,pred_clss #objs[B, np_max, 1],pred_boxes[B, np_max, 4],pred_clss[B, np_max, nc]
            return preds #, loss.sum() * B, loss.detach() # preds[nt, 4 + nc]
        else:
            objs = torch.sigmoid(objs) # objs[B, np_max, 1]
            mask_obj = objs[:,:,0] > self.thresh #mask_obj[B,np_max]
            objs = objs[mask_obj] #objs[mask_obj[B, np_max], 1] -> objs[np, 1(objs)]
            pred_boxes = pred_boxes[mask_obj] #pred_boxes[mask_obj[B, np_max], 4]->pred_boxes[np, 4]
            pred_clss = pred_clss[mask_obj] #pred_clss[mask_obj[B, np_max], nc]->pred_clss[np, nc]
            preds = torch.cat((objs,pred_boxes,pred_clss),dim=-1) #objs[np,1]+pred_boxes[np,4(box)]+pred_clss[np,nc]--> preds[np, 1(objs)+4(box)+nc]
            return preds #preds[np, 1(objs) + 4 + nc]

            
