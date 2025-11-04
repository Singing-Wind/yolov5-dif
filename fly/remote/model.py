import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vit_b_16,ViT_B_16_Weights
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from torchvision.models.vision_transformer import VisionTransformer
import math


class RemoteVitModel(nn.Module):
    def __init__(self,vit_head_out_num = 1024,dudv_size=32, hidden_size_xcyc=0, hidden_size_rrotation=128,C=256):
        super(RemoteVitModel, self).__init__()
        self.C = C
        self.keys = torch.linspace(0, 1, C, dtype=torch.float64)  # 0~1的C个关键值点
        ch_h = 3
        self.conv_in = nn.Conv2d(3, ch_h, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(ch_h)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv_in2 = nn.Conv2d(ch_h, 3, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(3)
        
        self.base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1) #was pretrained=True
        # self.vit_out_features = self.base_model.heads.head.out_features
        in_features = self.base_model.heads.head.in_features
        #self.vit_out_features -> vit_head_out_num
        self.base_model.heads.head = nn.Linear(in_features, vit_head_out_num) #1(obj)+2C(xc,yc)+1(theta)+1(dr)
        self.fc_out = nn.Linear(vit_head_out_num, 1+2*C+dudv_size) #输出维度变为2*C+1(obj)+dudv_size: obj(1), xc(C), yc(C), du(1), dv(1)

        # xcyc location Branch: xc_logits and yc_logits
        if hidden_size_xcyc>0:
            self.xcyc_branch = nn.Sequential(
                nn.Linear(2 * C, hidden_size_xcyc),
                nn.ReLU(),
                nn.Linear(hidden_size_xcyc, 2 * C)  # Outputting xc_logits and yc_logits
            )
        else:
            self.xcyc_branch = None

        # rotation & scale Branch: du and dv
        self.rrotate_branch = nn.Sequential(
            nn.Linear(dudv_size, hidden_size_rrotation),
            nn.ReLU(),
            nn.Linear(hidden_size_rrotation, 2)  # Outputting du and dv
        )

        #self.dropout = nn.Dropout(p=0.5)
        self.ln2 = math.log(1.2+0.001)

    def forward(self, x, alpha=0.5): #x[B,C,H,W]
        x = self.conv_in(x) #x[B,C=3,H,W]->x[B,ch_h,H,W]
        x = self.bn(x)

        # x = self.conv_in2(x) #x[B,C=3,H/2,W/2]
        # x = self.bn2(x) #x[B,C=3,H/2,W/2]
        
        # Resize input to 224x224 using bilinear interpolation #x[B,C,H,W]->x[B,C=3,H,W]
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)#->x[B,C=3,H=224,W=224]
        #
        x = self.base_model(x) #x[B,C=3,H=224,W=224]->x[B,vit_out_features]
        #
        y = self.fc_out(x)#x[B,vit_out_features]->x[B,2C+1+dudvsize=1(obj)+2C(xc,yc)+dudvsize]
        # 拆分输出
        obj = y[:, 0]  # [B]
        obj = torch.sigmoid(obj)
        #
        # Extract remaining outputs (excluding obj)
        y_remaining = y[:, 1:]  # [B, 2C + 2]

        # xcyc Branch
        if self.xcyc_branch != None:
            xcyc_output = self.xcyc_branch(y_remaining[:,:2*self.C])  # [B, 2C]
            xc_logits = xcyc_output[:, :self.C]  # [B, C]
            yc_logits = xcyc_output[:, self.C:]  # [B, C]
        else:
            xc_logits = y_remaining[:, :self.C]   # [B, C]
            yc_logits = y_remaining[:, self.C:2*self.C] # [B, C]

        # Rotation and Scale Branch
        rrot_output = self.rrotate_branch(y_remaining[:,2*self.C:])  # [B, 2]
        dtheta = rrot_output[:, 0]  # [B]
        dr = torch.exp(alpha * rrot_output[:, 1])  # [B]
        du = dr * torch.cos(dtheta)  # du[B]
        dv = dr * torch.sin(dtheta)  # dv[B]
        
        # xc = torch.sigmoid(xc)
        # yc = torch.sigmoid(yc)
        # du = torch.tanh(du) * 1.2
        # dv = torch.tanh(dv) * 1.2
        return obj, xc_logits, yc_logits, du, dv, self.keys.to(y.device) #keys[C]


    def predict(self, images):
        outputs = self(images) #obj_pred[B] xc_logits[B,C], yc_logits[B,C] du_pred[B], dv_pred[B] keys[C]

        obj_pred, xc_logits, yc_logits, du_pred, dv_pred, keys = outputs
        
        # 使用 dfl_pred 函数将 logits 转换为预测坐标
        xc_pred = dfl_pred(xc_logits, keys)# xc_logits[B,C],keys[C]->xc_pred[B]
        yc_pred = dfl_pred(yc_logits, keys)# yc_logits[B,C],keys[C]->yc_pred[B]

        return obj_pred,xc_pred,yc_pred, du_pred,dv_pred
    

def dfl_pred(logits, keys): #logits[B,C],keys[C] -> pred_val[B]
    # 对logits进行softmax得到概率分布
    probs = F.softmax(logits, dim=1)  # logits[B, C]->probs[B, C]
    # 目标分布在 idl, idr 两个位置有非零概率
    # p_idl = probs[torch.arange(B), idl] # probs[[B], idl[B]]->p_idl[B]
    # p_idr = probs[torch.arange(B), idr] # probs[[B], idr[B]]->p_idr[B]
    # 根据概率分布计算预测值
    pred_val = (probs * keys.unsqueeze(0)).sum(dim=1)  #probs[B][C] * keys[C]->[B][C] .sum(1)->pred_val[B]
    return pred_val #pred_val[B]