
import math
import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_
from .conv import Conv, DWConv
from .yolo_base import DetectDFL,OBB
import numpy as np
import os

# from ultralytics.GPT.embd import prompt2idx

class YoloText(DetectDFL):
    """YOLO YoloText detection head for detection with rotation models."""

    def __init__(self, nc=1, legacy=True, TMax=10, n_embd=512, ch=(), stride=()): # nc=80, legacy=True, ch=(), stride=()
        # Initialize YoloText with number of classes `nc` and layer channels `ch`.
        super().__init__(nc, legacy, ch, stride)
        #
        self.n_embd = n_embd

        self.TMax = TMax
        self.ne = TMax*self.n_embd  # number of extra parameters  改成self.ne = 0就恢复了

        c4 = 4 * max(ch)
        self.c4 = c4 #self.c4 = max(ch[0] // 4, self.ne)
        #self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 1)) for x in ch)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, self.ne, 1)) for x in ch)
        #self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), Conv(c4, self.ne, 1)) for x in ch)
        # self.cv4 = nn.ModuleList(nn.Sequential(nn.Conv2d(x, c4, 1)) for x in ch)
        # 初始化操作...
        # for param in self.cv4.parameters():
        #     param.requires_grad = False  # 冻结 cv4 层的参数
        # self.fc = nn.Linear(self.c4, self.ne, bias=False)
        # self.enable_fc = 1
        # if self.enable_fc:
        #     self.fc = nn.Sequential(
        #         nn.Linear(self.c4, self.c4, bias=False),
        #         nn.Linear(self.c4, self.c4, bias=False),
        #         nn.Linear(self.c4, self.ne, bias=False)) #ne=TMax*n_embd
        # else:
        #     self.fc = nn.Sequential(
        #         nn.Conv1d(self.c4, self.c4, kernel_size=1, bias=False),
        #         nn.Conv1d(self.c4, self.c4, kernel_size=1, bias=False),
        #         nn.Conv1d(self.c4, self.ne, kernel_size=1, bias=False) #ne=TMax*n_embd
        #     )
        # 1.Transformer 模块：输入 [mnt, TMax, n_embd]（其中 mnt 为前景目标数量）
        # 使用 batch_first=True 可以直接处理形状 [batch, seq, d_model]
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_embd, nhead=16, batch_first=True)
        self.text_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        if hasattr(self,'vocab_size'):
            # 2.文本 head：将 transformer 输出从 n_embd 映射到词汇表大小（50257）
            self.text_head = nn.Linear(self.n_embd, self.vocab_size)
        else:
            self.text_head = None

    def forward(self, x):#x[3][B,C,H,W]
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        if self.ne>0:#False and 
            ytext = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2) #ytext[b,ne=TMax*n_embd,ntotal=h0*w0+h1*w1+h2*w2]
            # ytext = ytext.detach()
        else: #only for debug
            ntotal = x[0].shape[-1]*x[0].shape[-2] + x[1].shape[-1]*x[1].shape[-2] + x[2].shape[-1]*x[2].shape[-2]
            ytext = torch.zeros((bs,self.ne,ntotal),dtype=x[0].dtype,device=x[0].device)
        assert ytext.shape==(bs,self.ne,x[0].shape[-1]*x[0].shape[-2] + x[1].shape[-1]*x[1].shape[-2] + x[2].shape[-1]*x[2].shape[-2])
        # 使用 detach() 以防止 ytext 分支影响主分支
        ytext = ytext.transpose(2, 1)#ytext[b,ne=TMax*n_embd,ntotal]->ytext[b,ntotal,ne=TMax*n_embd]
        # # NOTE: set `ytext` as an attribute so that `decode_text` could use it.
        # if not self.training:
        #     self.ytext = ytext #ytext[b,TMax*n_embd,ntotal=h0*w0+h1*w1+h2*w2]
        y = DetectDFL.forward(self, x) #y[3][B,16*4(box)+ncls,H,W]]  inference:y[b,4(box)+cls,ntotal],x  refer to DetectDFL::_inference
        if self.training:
            return y, ytext #y[3][B,16*4(box)+ncls,H,W]] ytext[B,ne,ntotal]
        #y[b,ntotal,4(box)+cls] + ytext[b,ntotal,TMax*n_embdl]->y[b,ntotal,4(box)+cls+TMax*n_embd]
        return (torch.cat([y[0], ytext], -1), y[1]) #interface match with DetectDFL
    
    def pred_text(self,rt):#rt[nt,nm=TMax*n_embd]
        if hasattr(self,'text_transformer') and next(self.text_transformer.parameters()).dtype==torch.float16:
            rt = rt.half()
        # y = self.fc(rt.half() if next(self.fc.parameters()).dtype==torch.float16 else rt) # ytext[mnt, TMax*n_embd] --> [mnt, ne=TMax*n_embd]
        nt = rt.shape[0]
        rt = rt.view(nt,self.TMax,self.n_embd)
        # Transformer 处理：得到新的文本特征 [mnt, TMax, n_embd]
        y = self.text_transformer(rt) #->y[mnt, TMax, n_embd]
        return y #[mnt, TMax,n_embd]


class OBBText(OBB):
    """YOLO YoloText detection head for detection with rotation models."""

    def __init__(self, nc=1, legacy=True, TMax=10, n_embd=512, ch=(), stride=()): # nc=80, legacy=True, ch=(), stride=()
        """Initialize YoloText with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, legacy, ch, stride)
        #
        self.ne = 1  # number of extra parameters  改成self.ne = 0就恢复了

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

        self.n_embd = n_embd
        self.TMax = TMax
        ctext = 4 * max(ch)
        self.ctext = ctext
        self.cvtext = nn.ModuleList(nn.Sequential(Conv(x, ctext, 3), Conv(ctext, TMax*self.n_embd, 1)) for x in ch)

        # 1.Transformer 模块：输入 [mnt, TMax, n_embd]（其中 mnt 为前景目标数量）
        # 使用 batch_first=True 可以直接处理形状 [batch, seq, d_model]
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_embd, nhead=16, batch_first=True)
        self.text_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        if hasattr(self,'vocab_size'):
            # 2.文本 head：将 transformer 输出从 n_embd 映射到词汇表大小（50257）
            self.text_head = nn.Linear(self.n_embd, self.vocab_size)
        else:
            self.text_head = None

    def forward(self, x):#x[3][B,C,H,W]
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        TMax_nembd = self.TMax*self.n_embd
        if TMax_nembd>0:#False and 
            ytext = torch.cat([self.cvtext[i](x[i]).view(bs, TMax_nembd, -1) for i in range(self.nl)], 2) #ytext[b,ne=TMax*n_embd,ntotal=h0*w0+h1*w1+h2*w2]
            # ytext = ytext.detach()
        else: #only for debug
            ntotal = x[0].shape[-1]*x[0].shape[-2] + x[1].shape[-1]*x[1].shape[-2] + x[2].shape[-1]*x[2].shape[-2]
            ytext = torch.zeros((bs,TMax_nembd,ntotal),dtype=x[0].dtype,device=x[0].device)
        assert ytext.shape==(bs,TMax_nembd,x[0].shape[-1]*x[0].shape[-2] + x[1].shape[-1]*x[1].shape[-2] + x[2].shape[-1]*x[2].shape[-2])
        # 使用 detach() 以防止 ytext 分支影响主分支
        ytext = ytext.transpose(2, 1)#ytext[b,ne=TMax*n_embd,ntotal]->ytext[b,ntotal,ne=TMax*n_embd]
        # # NOTE: set `ytext` as an attribute so that `decode_text` could use it.
        # if not self.training:
        #     self.ytext = ytext #ytext[b,TMax*n_embd,ntotal=h0*w0+h1*w1+h2*w2]
        y = OBB.forward(self, x) #y[3][B,16*4(box)+ncls,H,W]]  inference:y[b,4(box)+cls,ntotal],x  refer to OBB::_inference
        if self.training:
            return y, ytext #y[3][B,16*4(box)+ncls,H,W]] ytext[B,ne,ntotal]
        #y[b,ntotal,4(box)+cls] + ytext[b,ntotal,TMax*n_embd]->y[b,ntotal,4(box)+cls+TMax*n_embd]
        return (torch.cat([y[0], ytext], -1), y[1]) #interface match with OBB
    
    def pred_text(self,rt):#rt[nt,nm=TMax*n_embd]
        if hasattr(self,'text_transformer') and next(self.text_transformer.parameters()).dtype==torch.float16:
            rt = rt.half()
        # y = self.fc(rt.half() if next(self.fc.parameters()).dtype==torch.float16 else rt) # ytext[mnt, TMax*n_embd] --> [mnt, ne=TMax*n_embd]
        nt = rt.shape[0]
        rt = rt.view(nt,self.TMax,self.n_embd)
        # Transformer 处理：得到新的文本特征 [mnt, TMax, n_embd]
        y = self.text_transformer(rt) #->y[mnt, TMax, n_embd]
        return y #[mnt, TMax,n_embd]