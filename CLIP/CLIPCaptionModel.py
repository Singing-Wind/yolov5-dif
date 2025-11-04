import os

import torch
from torch import nn
from .space_to_depth import space_to_depth

from transformers import (
    CLIPModel, CLIPProcessor,
    GPT2Tokenizer, GPT2LMHeadModel
)
# pip install torch torchvision transformers datasets tqdm
from torch.optim import AdamW
from transformers import AutoConfig
import logging
from transformers.utils import logging as hf_logging

from torchvision.transforms.functional import to_pil_image

hf_logging.set_verbosity_error()

class CLIPCaptionModel(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", gpt2_model_name="gpt2", block_size=4):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.block_size = block_size
        self.Np = block_size * block_size
        self.clip_feature_size = 512
        self.clip_proj = nn.Linear(self.clip_feature_size, self.gpt2.config.n_embd)

    def forward(self, images, captions):
        device = next(self.parameters()).device

        if self.block_size>1:# 1. 将图像切分成多个 patch
            patches = space_to_depth(images,self.block_size)  # [B, Np, C, H//bs, W//bs]
            B, Np, C, H_patch, W_patch = patches.shape
            patches = patches.view(B * Np, C, H_patch, W_patch) #patches[B*Np,C=3,Hs,Ws]
        else:
            B, C, H_patch, W_patch = images.shape
            Np = 1
            patches = images

        # 1. 图像特征
        inputs = self.processor(images=patches, return_tensors="pt", padding=True, do_rescale=False).to(device)
        with torch.no_grad():
            image_features = self.clip.get_image_features(**inputs) #image_features[B*Np,512]
        image_features = image_features.view(B, Np, self.clip_feature_size)  # [B, Np, 512]
        prefix_embed = self.clip_proj(image_features)  #prefix_embed[B, Np, n_embd=768]

        # 2. 文本 token 化
        tokenized = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokenized.input_ids.to(device) #input_ids[B,ntext]
        labels = input_ids.clone()#labels[B,ntext]
        gpt_embeds = self.gpt2.transformer.wte(input_ids)#gpt_embeds[B,ntext,n_embd=768]

        # 3.拼接图像前缀和GPT文本输入 prefix_embed[B, Np, n_embd=768],gpt_embeds[B,Lc,n_embd=768]->inputs_embeds[b,Np+Lc,n_embd=768]
        inputs_embeds = torch.cat((prefix_embed, gpt_embeds), dim=1) #inputs_embeds[b,Np+Lc,n_embd=768]
        labels = torch.cat((torch.full((labels.size(0), prefix_embed.shape[-2]), -100).to(device), labels), dim=1) #(-100)[b,Np]+labels[b,Lc]->labels[b,Np(-100)+Lc] -100 is for no loss

        output = self.gpt2(inputs_embeds=inputs_embeds, labels=labels) #inputs_embeds[b,Np(image nembd)+Lc,n_embd=768] -> labels[b,Np(-100)+Lc]
        return output.loss


class CLIPPromptedCaptionModel(CLIPCaptionModel):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", gpt2_model_name="gpt2", block_size=4):
        super().__init__(clip_model_name, gpt2_model_name, block_size)

    def forward(self, images, prompts, captions):
        # images[B,C=3,H,W]
        # prompts[B]['',...]
        # captions[B]['',...] : list of str, e.g. ["a dog running in the park", "a mountain with snow"]
        device = next(self.parameters()).device

        if self.block_size>1:# 1. 将图像切分成多个 patch
            patches = space_to_depth(images,self.block_size)  # [B, Np, C, H//bs, W//bs]
            B, Np, C, H_patch, W_patch = patches.shape
            patches = patches.view(B * Np, C, H_patch, W_patch) #patches[B*Np,C=3,Hs,Ws]
        else:
            B, C, H_patch, W_patch = images.shape
            Np = 1
            patches = images

        # 1. 图像特征
        inputs = self.processor(images=patches, return_tensors="pt", padding=True, do_rescale=False).to(device)
        with torch.no_grad():
            image_features = self.clip.get_image_features(**inputs) # image_features[B*Np, 512]
        image_features = image_features.view(B, Np, self.clip_feature_size)  # [B, Np, 512]
        prefix_embed = self.clip_proj(image_features)  # [B, Np, n_embd=768]

        # 2. 提示词 + captions 组合（拼接后再 token 化也可以，但我们按块分开处理更灵活）
        prompt_tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        prompt_ids = prompt_tokenized.input_ids.to(device).to(torch.int64) #prompt_ids[B,Lp]
        prompt_embeds = self.gpt2.transformer.wte(prompt_ids)  # [B, prompt_len, n_embd=768]

        caption_tokenized = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
        caption_ids = caption_tokenized.input_ids.to(device).to(torch.int64)
        caption_embeds = self.gpt2.transformer.wte(caption_ids)  # [B, caption_len, n_embd=768]

        # 拼接: [prefix, prompt, caption]
        inputs_embeds = torch.cat((prefix_embed, prompt_embeds, caption_embeds), dim=1) #prefix_embed[B,Np,n_embd=768]+prompt_embeds[B,Lp,n_embd=768]+caption_embeds[B,Lc,n_embd=768]-->[B,Np+Lp+Lc,n_embd=768]

        # 构建 labels：prefix 和 prompt 部分不算 loss
        ignore_prefix_prompt = torch.full((caption_ids.size(0), prefix_embed.shape[-2] + prompt_ids.size(1)), -100).to(device) #ignore_prefix_prompt[B,Np+Lp]
        labels = torch.cat((ignore_prefix_prompt, caption_ids), dim=1) #ignore_prefix_prompt[B,Np+Lp]+caption_ids[B,Lc]->labels[B,Np+Lp+Lc]

        output = self.gpt2(inputs_embeds=inputs_embeds, labels=labels)
        return output.loss

    @torch.no_grad()
    def generate(self, image, prompt, max_new_tokens=50):
        # image: a single image (PIL or tensor)
        # prompt: str, textual hint (e.g. "A picture of")
        device = next(self.parameters()).device

        inputs = self.processor(images=image, return_tensors="pt", padding=True, do_rescale=False).to(device)
        image_features = self.clip.get_image_features(**inputs) #image_features[1,512]
        prefix_embed = self.clip_proj(image_features).unsqueeze(1) #image_features[1,512]->[1, 768]->prefix_embed[1, 1, 768]

        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device).to(torch.int64)
        prompt_embeds = self.gpt2.transformer.wte(prompt_ids)  # prompt_embeds[1, L, 768]

        inputs_embeds = torch.cat((prefix_embed, prompt_embeds), dim=1) #prefix_embed[1,1,768]+prompt_embeds[1,Lp,768]->inputs_embeds[1,1+Lp,768]

        output_ids = self.gpt2.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
