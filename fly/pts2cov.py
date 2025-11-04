import cv2
import numpy as np
import torch
import math

def pts2cov(pts,gauss=0):  # n * 8
    # pts -> (cx, cy, a, b, cos,sin)
    n = pts.shape[0]
    if n:
        # 目标自身的宽高
        cx = pts[:, ::2].sum(dim=1) / 4
        cy = pts[:, 1::2].sum(dim=1) / 4
        xf, yf = (pts[:, 0] + pts[:, 2]) / 2, (pts[:, 1] + pts[:, 3]) / 2
        xb, yb = (pts[:, 4] + pts[:, 6]) / 2, (pts[:, 5] + pts[:, 7]) / 2
        dx = xf - xb
        dy = yf - yb
        L2 = dx ** 2 + dy ** 2
        L = torch.sqrt(L2)
        # 方向
        cos = dx / L
        sin = dy / L

        # cx = (pts[:, 0] + pts[:, 2] + pts[:, 4] + pts[:, 6]) / 4
        # cy = (pts[:, 1] + pts[:, 3] + pts[:, 5] + pts[:, 7]) / 4
        txy = torch.zeros_like(pts, dtype=torch.float)
        px = pts[:, ::2] - cx[:, None]
        py = pts[:, 1::2] - cy[:, None]
        txy[:, ::2] = cos[:, None] * px + sin[:, None] * py
        txy[:, 1::2] = -sin[:, None] * px + cos[:, None] * py
        lx = (txy[:, 4] + txy[:, 6]) / 2
        rx = (txy[:, 0] + txy[:, 2]) / 2
        a = (rx - lx) / 2
        ty = (txy[:, 1] + txy[:, 7]) / 2
        by = (txy[:, 3] + txy[:, 5]) / 2
        b = (by - ty) / 2
        # assert (a > 0).any()
        # assert (b > 0).any()
        
        if gauss==0:
            out = torch.stack((cx, cy, a, b, cos,sin), dim=1)
        else:
            cos2 = cos.pow(2)
            sin2 = sin.pow(2)
            a, b = a ** 2 / 12, b ** 2 / 12
            a, b, c = a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin
            # c = torch.atan2(sin,cos)
            out = torch.stack((cx, cy, a, b, c), dim=1)
    else:
        out = torch.zeros(0, 6 if gauss==0 else 5, dtype=torch.float32).to(pts.device)
    return out

def obb2gauss(obb):
    assert obb.shape[-1]==6
    xy = obb[:,0:2]#xy[np,2]
    ab = obb[:,2:4]#ab[np,2]
    cs = obb[:,4:6]#cs[np,2]
    cos,sin = cs[:,0],cs[:,1]
    c2s2 = cs.pow(2)#c2s2[np,2]
    cos2,sin2 = c2s2[:,0],c2s2[:,1]
    ab = ab ** 2 / 12 #ab[np,2]
    a,b = ab[:,0],ab[:,1]
    a, b, c = a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin
    obb = torch.cat([xy,a[:,None],b[:,None],c[:,None]],dim=1).clone()
    return obb
def probiou_fly_gauss(obb1, obb2, eps=1e-7):
    if obb1.shape[-1]!=5:
        obb1 = obb2gauss(obb1).clone()
    if obb2.shape[-1]!=5:
        obb2 = obb2gauss(obb2).clone()
    #
    x1, y1, a1, b1, c1 = obb1.split(1, dim=-1)
    x2, y2, a2, b2, c2 = (x.squeeze(-1)[None] for x in obb2.split(1, dim=-1))
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    return iou

def obb_to_polygon_np(obb):
    # 输入：obb [N,6] -> x,y,a,b,cos,sin
    # 输出：[N,4,2] 顶点坐标
    x, y, a, b, c, s = [obb[:, i] for i in range(6)]
    dx1 = c * a
    dy1 = s * a
    dx2 = -s * b
    dy2 = c * b
    p1 = np.stack([x - dx1 - dx2, y - dy1 - dy2], axis=-1)
    p2 = np.stack([x + dx1 - dx2, y + dy1 - dy2], axis=-1)
    p3 = np.stack([x + dx1 + dx2, y + dy1 + dy2], axis=-1)
    p4 = np.stack([x - dx1 + dx2, y - dy1 + dy2], axis=-1)
    return np.stack([p1, p2, p3, p4], axis=1)  # [N, 4, 2]
def probiou_fly(obb1, obb2, eps=1e-7):
    assert obb1.shape[-1]==6 and obb2.shape[-1]==6
    # obb1: [N1, 6], obb2: [N2, 6]
    # return: [N1, N2] iou matrix
    obb1 = obb1.cpu().numpy() if isinstance(obb1, torch.Tensor) else obb1
    obb2 = obb2.cpu().numpy() if isinstance(obb2, torch.Tensor) else obb2

    poly1 = obb_to_polygon_np(obb1)  # [N1, 4, 2]
    poly2 = obb_to_polygon_np(obb2)  # [N2, 4, 2]
    
    N1, N2 = poly1.shape[0], poly2.shape[0]
    iou = np.zeros((N1, N2), dtype=np.float32)

    for i in range(N1):
        p1 = poly1[i].astype(np.float32)
        area1 = cv2.contourArea(p1)

        if area1 < eps:
            continue

        for j in range(N2):
            p2 = poly2[j].astype(np.float32)
            area2 = cv2.contourArea(p2)

            if area2 < eps:
                continue

            inter_area, _ = cv2.intersectConvexConvex(p1, p2)
            union_area = area1 + area2 - inter_area
            iou[i, j] = inter_area / (union_area + eps)

    return torch.from_numpy(iou)

def cs2arcarrow2(cs,ang):#cs[np,2]->[np,4]
    cosd,sind = math.cos(ang),math.sin(ang)
    cos1,sin1 = cs[:,0]*cosd-cs[:,1]*sind, cs[:,0]*sind+cs[:,1]*cosd
    cos2,sin2 = cs[:,0]*cosd+cs[:,1]*sind,-cs[:,0]*sind+cs[:,1]*cosd
    return torch.stack([cos1, sin1, cos2, sin2], dim=-1)