import torch
import torch.nn as nn
import torch.nn.functional as F

class LocAtt(nn.Module):
    def __init__(self, in_channels, C, kernel_size=3):
        super(LocAtt, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.to_q = nn.Conv2d(in_channels, C, kernel_size=1)
        self.to_k = nn.Conv2d(in_channels, C, kernel_size=1)
        self.to_v = nn.Conv2d(in_channels, C, kernel_size=1)

    def forward(self, x):# x[B, in_channels, H, W]
        B, C, H, W = x.size()
        q = self.to_q(x)  # [B, C, H, W]
        k = self.to_k(x)  # [B, C, H, W]
        v = self.to_v(x)  # [B, C, H, W]

        # unfold k,v: shape becomes [B, C * K*K, H*W]
        k_unf = F.unfold(k, kernel_size=self.kernel_size, padding=self.pad)  # [B, C*K*K, H*W]
        v_unf = F.unfold(v, kernel_size=self.kernel_size, padding=self.pad)  # [B, C*K*K, H*W]

        # reshape to [B, C, K*K, H*W]
        k_unf = k_unf.view(B, C, self.kernel_size**2, H*W)
        v_unf = v_unf.view(B, C, self.kernel_size**2, H*W)

        # reshape q to [B, C, H*W]
        q_flat = q.view(B, C, H*W).unsqueeze(2)  # [B, C, 1, H*W]

        # attention scores: dot(q, k)
        attn = (q_flat * k_unf).sum(dim=1)  # [B, K*K, H*W]

        # softmax over kernel dimension
        attn = F.softmax(attn, dim=1)  # [B, K*K, H*W]

        # weighted sum: [B, C, H*W]
        y = (attn.unsqueeze(1) * v_unf).sum(dim=2)  # [B, C, H*W]

        # reshape back to [B, C, H, W]
        y = y.view(B, C, H, W)
        return y #y[B, C, H, W]


class LocAtt2(nn.Module):
    def __init__(self, in_channels, C, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.q_proj = nn.Conv2d(in_channels, C, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, C, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, C, kernel_size=1)

    def forward(self, x): #x[B, in_channels, H, W]
        B, C, H, W = x.shape

        # 1. 计算 Q, K, V
        q = self.q_proj(x)                        # [B, C, H, W]
        k = self.k_proj(x)                        # [B, C, H, W]
        v = self.v_proj(x)                        # [B, C, H, W]

        # 2. 展开 K, V 的局部邻域 (3x3)
        k_unf = F.unfold(k, kernel_size=self.kernel_size, padding=self.padding)  # [B, C * 9, H*W]
        v_unf = F.unfold(v, kernel_size=self.kernel_size, padding=self.padding)  # [B, C * 9, H*W]

        k_unf = k_unf.view(B, C, self.kernel_size**2, H, W)  # [B, C, 9, H, W]
        v_unf = v_unf.view(B, C, self.kernel_size**2, H, W)  # [B, C, 9, H, W]

        # 3. 点积注意力
        q = q.unsqueeze(2)                       # [B, C, 1, H, W]
        attn = (q * k_unf).sum(dim=1)            # [B, 9, H, W]
        attn = F.softmax(attn, dim=1)            # [B, 9, H, W]

        # 4. 对 V 做加权求和
        out = (attn.unsqueeze(1) * v_unf).sum(dim=2)  # [B, C, H, W]

        return out #out[B, C, H, W]
    

class LocAttGrouped(nn.Module):
    def __init__(self, in_channels, C, kernel_size=3, num_groups=4):
        super(LocAttGrouped, self).__init__()
        assert C % num_groups == 0, "C must be divisible by num_groups"
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.num_groups = num_groups
        self.group_dim = C // num_groups

        self.to_q = nn.Conv2d(in_channels, C, kernel_size=1)
        self.to_k = nn.Conv2d(in_channels, C, kernel_size=1)
        self.to_v = nn.Conv2d(in_channels, C, kernel_size=1)

    def forward(self, x):  # x: [B, in_channels, H, W]
        B, _, H, W = x.size()
        C = self.group_dim * self.num_groups

        q = self.to_q(x)  # q[B, C, H, W]
        k = self.to_k(x)  # k[B, C, H, W]
        v = self.to_v(x)  # v[B, C, H, W]

        # reshape to [B, G, group_dim, H, W]
        q = q.view(B, self.num_groups, self.group_dim, H, W)
        k = k.view(B, self.num_groups, self.group_dim, H, W)
        v = v.view(B, self.num_groups, self.group_dim, H, W)

        # apply unfold on k,v: shape [B, G, group_dim * K*K, H*W]
        k_unf = F.unfold(k.view(B * self.num_groups, self.group_dim, H, W),
                         kernel_size=self.kernel_size, padding=self.pad)
        v_unf = F.unfold(v.view(B * self.num_groups, self.group_dim, H, W),
                         kernel_size=self.kernel_size, padding=self.pad)

        k_unf = k_unf.view(B, self.num_groups, self.group_dim, self.kernel_size**2, H*W)
        v_unf = v_unf.view(B, self.num_groups, self.group_dim, self.kernel_size**2, H*W)

        # reshape q: [B, G, group_dim, 1, H*W]
        q = q.view(B, self.num_groups, self.group_dim, H*W).unsqueeze(3)

        # attention logits: [B, G, K*K, H*W]
        attn = (q * k_unf).sum(dim=2)  # dot product over group_dim

        attn = F.softmax(attn, dim=2)  # softmax over kernel window dim

        # apply attention weights to v_unf
        out = (attn.unsqueeze(2) * v_unf).sum(dim=3)  # [B, G, group_dim, H*W]

        # reshape back to [B, C, H, W]
        out = out.view(B, C, H, W)
        return out #out: [B, C, H, W]