import torch
import torch.nn as nn
from models.common import *
from utils.tal import dist2bbox, dist2rbox
from .Transformer import TransformerMLPHead

def sauv2A23(stride,shape,s,angle,du,dv):
    # NOTE: ne=4(saxy)  scale,angle,dx,dy
    imgsz = [stride*shape[-2],stride*shape[-1]] #imgsz[H,W]
    dx = du * imgsz[1]#dx
    dy = dv * imgsz[0]#dy
    xd,yd = imgsz[1] / 2, imgsz[0] / 2

    cos_rad,sin_rad = torch.cos(angle),torch.sin(angle) #[b]
    a,b = s*cos_rad, s*sin_rad#[b]
    cx,cy = xd + dx, yd + dy#[b]
    tx = xd - s*( cos_rad*cx+sin_rad*cy)#[b]
    ty = yd - s*(-sin_rad*cx+cos_rad*cy)#[b]

    # A23 = torch.tensor([[a,b, tx],[-b,a,ty]],dtype=torch.float16)
    # 创建 A23 张量，形状 [n, 2, 3]
    A23s = torch.stack([
        torch.stack([ a, b, tx], dim=1),
        torch.stack([-b, a, ty], dim=1)
    ], dim=1) #A23[B,2,3]
    return A23s

def abuv2A23s(imgsz,a,b,du,dv):
    # NOTE: ne=4(saxy)  scale,angle,dx,dy
    dx = du
    dy = dv
    xd,yd = imgsz[1] / 2, imgsz[0] / 2

    cx,cy = xd + dx, yd + dy#[b]
    tx = xd - ( a*cx + b*cy)#
    ty = yd - (-b*cx + a*cy)#

    # A23 = torch.tensor([[a,b, tx],[-b,a,ty]],dtype=torch.float16)
    # 创建 A23 张量，形状 [n, 2, 3]
    A23s = torch.stack([
        torch.stack([ a, b, tx], dim=1),
        torch.stack([-b, a, ty], dim=1)
    ], dim=1) #A23[B,2,3]
    return A23s
def A23s2abuv(imgsz,A23s):#imgsz = [stride*shape[-2],stride*shape[-1]] #imgsz[H,W]
    # NOTE: ne=4(saxy)  scale,angle,dx,dy
    xd,yd = imgsz[1] / 2, imgsz[0] / 2

    a,b = A23s[:,0,0],A23s[:,0,1]  
    tx,ty = A23s[:,0,2],A23s[:,1,2]
    detx,dety = xd-tx,yd-ty #( a*cx + b*cy),(-b*cx + a*cy)
    detab = a*a+b*b
    xcd,ycd = (a*detx - b* dety)/detab, (b*detx + a*dety)/detab
    du,dv = xcd - xd, ycd - yd

    abuv = torch.stack([a,b,du,dv], dim=1) #abuv[bs,4(abuv)]

    return abuv #abuv[bs,4(abuv)]

class HA23(nn.Module):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, m, max_dudv, ch=(), stride=()): #m=64, max_dudv=100.0
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__()
        self.m = m
        self.keys = torch.linspace(-max_dudv, max_dudv, m, dtype=torch.float64) #keys[m]
        self.ne = 4 # number of extra parameters,输出通道数量（a,b，du,dv）  （s,θ，dx,dy）
        self.mne = 2 + m * 2 #2(ab)+2(dudv)m
        c4 = max(ch[0] // 4, self.ne)
        self.c4 = c4
        # self.cv4 =nn.Sequential(Conv(in_ch, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1))
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, c4, 1)) for x in ch)
        self.stride = stride[-1]

        if 0:
            self.cv_uv = TransformerMLPHead(c4,256,m*2,4,4)
        else:
            self.cv_uv = nn.Linear(c4,m*2)
        self.cv_ab = nn.Linear(c4,2)
            # self.KA = nn.Linear(self.mne,self.ne)

    def forward(self, x_): #x[0][B,C=256,H,W], x[1] from OBB
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        x = x_[:-1] #x_[-1]是obb层的,剔除掉

        bs = x[0].shape[0]  # batch size
        ys = torch.cat([self.cv4[i](x[i]).view(bs, self.c4, -1) for i in range(len(x))], 2)#ys[b,c4,h*w]->ys[b,c4,ntotal=h0*w0+h1*w1+h2*w2]
        ys = ys.mean(dim=-1) #ys[b,c4,ntotal=h0*w0+h1*w1+h2*w2]->ys[b,c4]

        #ab
        ab = self.cv_ab(ys) #ab[b,2(ab)]

        # y = self.transformer(ys) #ys[b,c4]->y[b,4(abuv)]
        quv = self.cv_uv(ys).view(bs,2,self.m) #ys[b,c4]->quv[b,2(uv)*m]->quv[b,2(uv),m]
        assert quv.shape[0]==bs
        puv = torch.softmax(quv,dim=-1) #puv[b,2(uv),m]
        if self.keys.device!=puv.device:
            self.keys = self.keys.to(device=puv.device)
        duv = (puv * self.keys.view(1,1,self.m)).sum(-1) #p[b,2(uv),m] * keys[1,1,m] -> [b,2(uv),m] -> y[b,2(uv)]
        #

        y = torch.cat([ab,duv],dim=1) #ab[b,2(ab)],duv[b,2(uv)] -> y[b,4(abuv)]

        return y, x_[-1] #y[b,4(abuv)]  x_[-1](from obb)

    def abuv2A23s(self,imgsz, y):
        #abuv->A23
        a = y[:,0] #a[bs]
        b = y[:,1] #b[bs]
        du = y[:,2]#du[bs]
        dv = y[:,3]#dv[bs]

        # imgsz = [self.stride*x[0].shape[-2],self.stride*x[0].shape[-1]] #imgsz[H,W]
        A23s = abuv2A23s(imgsz,a,b,du,dv) #A23s[b,2,3]
  
        return A23s # A23.view(b,2,3)
