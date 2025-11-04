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

from .DFL import dfl_pred

from models.common import *

class VitA23(nn.Module): #vit_head_out_num = 1024,dudv_size=32, hidden_size_xcyc=64, hidden_size_rrotation=128,C=256, max_dudv=128
    def __init__(self,vit_head_out_num,dudv_size, hidden_size_xcyc, hidden_size_rrotation,C, max_dudv, ch=(), stride=()):
        super(VitA23, self).__init__()
        c4 = ch[0] // 4
        self.c4 = c4
        # self.cv4 =nn.Sequential(Conv(in_ch, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1))
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, c4, 1)) for x in ch)
        self.stride = stride[-1]
        
        self.C = C
        self.keys = torch.linspace(-max_dudv, max_dudv, C, dtype=torch.float64)  # 0~1的C个关键值点
        ch_h = 3
        self.conv_in = nn.Conv2d(self.c4, ch_h, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(ch_h)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv_in2 = nn.Conv2d(ch_h, 3, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(3)
        
        self.base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1) #was pretrained=True
        # self.vit_out_features = self.base_model.heads.head.out_features
        in_features = self.base_model.heads.head.in_features
        #self.vit_out_features -> vit_head_out_num
        self.base_model.heads.head = nn.Linear(in_features, vit_head_out_num) #1(obj)+2C(du,dv)+1(theta)+1(dr)
        self.fc_out = nn.Linear(vit_head_out_num, 2*C+dudv_size) #输出维度变为2*C+dudv_size: du(C), dv(C), a(1), b(1)

        # xcyc location Branch: xc_logits and yc_logits
        if hidden_size_xcyc>0:
            self.xcyc_branch = nn.Sequential(
                nn.Linear(2 * C, hidden_size_xcyc),
                nn.ReLU(),
                nn.Linear(hidden_size_xcyc, 2 * C)  # Outputting xc_logits and yc_logits
            )
        else:
            self.xcyc_branch = None

        # rotation & scale Branch: a and b
        self.rrotate_branch = nn.Sequential(
            nn.Linear(dudv_size, hidden_size_rrotation),
            nn.ReLU(),
            nn.Linear(hidden_size_rrotation, 2)  # Outputting a and b
        )

        #self.dropout = nn.Dropout(p=0.5)
        self.ln2 = math.log(1.2+0.001)

    def forward(self, x_, alpha=0.5): #x[B,C,H,W]
        x = x_[:-1] #x_[-1]是obb层的,剔除掉

        bs = x[0].shape[0]  # batch size
        # xs = torch.cat([self.cv4[0](x[i]).view(bs, self.c4, -1) for i in range(len(x))], 2)#xs[b,c4,h*w]->xs[b,c4,ntotal=h0*w0+h1*w1+h2*w2]
        # xs = xs.mean(dim=-1)
        xs = self.cv4[0](x[0]) #xs[b,c4,ntotal=h0*w0+h1*w1+h2*w2]->xs[b,c4]

        x = self.conv_in(xs) #xs[B,C=3,H,W]->x[B,ch_h,H,W]
        x = self.bn(x)

        # x = self.conv_in2(x) #x[B,C=3,H/2,W/2]
        # x = self.bn2(x) #x[B,C=3,H/2,W/2]
        
        # Resize input to 224x224 using bilinear interpolation #x[B,C,H,W]->x[B,C=3,H,W]
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)#->x[B,C=3,H=224,W=224]
        #
        x = self.base_model(x) #x[B,C=3,H=224,W=224]->x[B,vit_out_features]
        #
        y_remaining = self.fc_out(x)#x[B,vit_out_features]->y_remaining[B,2C+dudvsize = 2C(du,dv)+dudvsize]
        # 拆分输出

        if self.keys.device!=y_remaining.device:
            self.keys = self.keys.to(device=y_remaining.device)

        # xcyc Branch
        xcyc_output = self.xcyc_branch(y_remaining[:,:2*self.C])  # [B, 2C]
        xc_logits = xcyc_output[:, :self.C]  # [B, C]
        du = dfl_pred(xc_logits, self.keys) #xc_logits[B, C],keys[C]->du[B]
        yc_logits = xcyc_output[:, self.C:]  # [B, C]
        dv = dfl_pred(yc_logits, self.keys) #yc_logits[B, C],keys[C]->dv[B]
        
        # Rotation and Scale Branch
        rrot_output = self.rrotate_branch(y_remaining[:,2*self.C:])  # [B, 2]
        dtheta = rrot_output[:, 0]  # [B]
        dr = torch.exp(alpha * rrot_output[:, 1])  # [B]
        a = dr * torch.cos(dtheta)  # a[B]
        b = dr * torch.sin(dtheta)  # b[B]

        y = torch.cat([a.unsqueeze(1),b.unsqueeze(1),du.unsqueeze(1),dv.unsqueeze(1)],dim=1) #a[b,1],b[b,1],du[b,1],dv[b,1] -> y[b,4(abuv)]

        return y, x_[-1] #y[b,4(abuv)]  x_[-1](from obb)

