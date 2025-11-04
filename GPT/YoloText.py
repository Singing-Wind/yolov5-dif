import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class YOLOv5HeadWithText(nn.Module):
    def __init__(self, num_classes, nembd=256, max_len=20, voc_size=30000):
        super(YOLOv5HeadWithText, self).__init__()
        
        # 原YOLOv5输出的框坐标、类别和置信度
        self.num_classes = num_classes
        self.nembd = nembd  # 目标描述向量的维度
        self.max_len = max_len  # 最大文本长度
        
        # 原始的YOLOv5 head (框架保持不变)
        self.regression = nn.Conv2d(256, 4, kernel_size=1)  # 目标位置回归
        self.classification = nn.Conv2d(256, num_classes, kernel_size=1)  # 目标类别预测
        self.obj_confidence = nn.Conv2d(256, 1, kernel_size=1)  # 目标置信度
        
        # 新增的文字描述分支
        self.text_embedding = nn.Conv2d(256, nembd * max_len, kernel_size=1)  # 每个目标nembd维的文本表示
        self.text_transformer = GPT2LMHeadModel.from_pretrained("gpt2")  # 使用预训练的GPT模型进行文本生成
        
        # 词汇表大小，最后通过softmax输出词汇的概率分布
        self.voc_size = voc_size
        self.text_decoder = nn.Linear(nembd, voc_size)  # 将nembd映射为词汇表的大小
    
    def forward(self, x):
        # YOLOv5原本的head部分
        reg_output = self.regression(x)  # 目标框坐标
        cls_output = self.classification(x)  # 目标类别
        obj_output = self.obj_confidence(x)  # 目标置信度
        
        # 新增的文字描述部分
        text_emb = self.text_embedding(x).view(x.size(0), -1, self.nembd, self.max_len)  # [Batch, num_boxes, nembd, T]
        text_emb = text_emb.permute(0, 2, 1, 3)  # 转置后变为[Batch, nembd, num_boxes, T]
        
        # 通过transformer生成目标文字描述
        text_output = self.text_transformer(inputs_embeds=text_emb)  # [Batch, num_boxes, T, voc_size]
        
        # 对生成的文本进行解码
        text_logits = self.text_decoder(text_output.last_hidden_state)  # [Batch, num_boxes, T, voc_size]
        
        # 选择每个位置的最大值词汇作为目标描述
        text_pred = torch.argmax(text_logits, dim=-1)  # 获取最大概率的词，作为目标的描述
        
        return reg_output, cls_output, obj_output, text_pred



class YOLOv5WithTextLoss(nn.Module):
    def __init__(self, num_classes, lambda_gpt=1.0):
        super(YOLOv5WithTextLoss, self).__init__()
        self.lambda_gpt = lambda_gpt  # 调整文本生成loss的权重
        
        # 原YOLOv5的损失
        self.bce_loss = nn.BCEWithLogitsLoss()  # 置信度损失
        self.cce_loss = nn.CrossEntropyLoss()  # 类别交叉熵损失
        self.mse_loss = nn.MSELoss()  # 坐标回归损失

    def forward(self, pred, target):
        # 计算YOLOv5的损失
        reg_pred, cls_pred, obj_pred, text_pred = pred
        reg_target, cls_target, obj_target, text_target = target
        
        # YOLOv5的传统损失
        loss_reg = self.mse_loss(reg_pred, reg_target)  # 坐标损失
        loss_cls = self.cce_loss(cls_pred, cls_target)  # 类别损失
        loss_obj = self.bce_loss(obj_pred, obj_target)  # 置信度损失
        
        # GPT生成的文字描述损失
        loss_gpt = self.cce_loss(text_pred.view(-1, text_pred.size(-1)), text_target.view(-1))  # 文本生成损失
        
        # 最终的损失
        total_loss = loss_reg + loss_cls + loss_obj + self.lambda_gpt * loss_gpt
        
        return total_loss

