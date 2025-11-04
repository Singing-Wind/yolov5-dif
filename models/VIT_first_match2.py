
"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict
 
import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x, drop_prob:float=0., training: bool=False): # x:[b, c, H, W]
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    
    if drop_prob == 0. or not training:
        return x
    
    
    # 保留分支的概率
    keep_prob = 1. - drop_prob

    shape = (x.shape[0],) + (1,)*(x.ndim-1) # shape[bs, 1, 1, 1]， x.ndim返回x的维度
    keep_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device) # keep_tensor[bs, 1, 1, 1]
    keep_tensor.floor_() # binarize, floor_是floor的原位运算,向下取整。  keep_tensor如[1, 0, 1, 1, 1](drop_prob=0.2时). E(keep_tensor)=keep_prob

    output = x.div(keep_prob)*keep_tensor  # output:[b, c, H, W].   div部分是为了保持和原始输入 x 相同的期望.

    return output



class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)  #init中无需定义self.training
    


class PatchEmbed(nn.Module):
    '''2D Image to Patch Embedding. 接收img, 输出则作为Transformer Encoder的输入, 包含conv和flatten'''

    def __init__(self, img_size=640, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (self.img_size[0]//self.patch_size[0], self.img_size[1]//self.patch_size[1])
        self.in_c = in_c
        self.num_patches = self.grid_size[0] * self.grid_size[1]


        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    

    def forward(self, x):
        
        # x:img[B, C, H, W]
        _, C, H, W = x.shape
        # 传入的图片高和宽和预设的不一样就会报错,VIT模型里面输入的图像大小必须是固定的
        assert C==self.in_c and H==self.img_size[0] and W==self.img_size[1], f"Input img size ({C},{H},{W}) doesn't match model({self.in_c},{self.img_size[0]},{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1,2) # x[B, embed_dim=768, grid_size(h), grid_size(w)] -> x[b, 768, hw] -> x[b, hw, 768]
        x = self.norm(x)

        return x # x[b, hw, 768]


class Attention(nn.Module):

    def __init__(self, in_dim=768, num_heads=8, qk_scale=None, qkv_bias=False, attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()

        self.in_dim=in_dim
        self.num_heads = num_heads
        self.head_dim = self.in_dim//self.num_heads
        self.scale = qk_scale or (self.head_dim)**(-0.5) # 计算分母，q,k相乘之后要除以一个根号下dk。

        self.qkv = nn.Linear(in_dim, in_dim*3, bias=qkv_bias) #Linear函数可以接收多维的矩阵输入但是只对最后一维起效果，其他维度不变。
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    

    def forward(self, x, rotary_emb=None):
        
        # x[b, hw, in_dim=768],这里不是hw+1,因为我们的任务无需引入cls_embed
        B, hw, C = x.shape
        assert C==self.in_dim, f"图像embedding维度{C}与Attn实例化维度{self.in_dim}不符"
        qkv = self.qkv(x).reshape(B, hw, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)    # x[b, hw, in_dim*3] -> [3, b, n_head, hw, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2] # q,k,v: [b, n_head, hw, c_per_head]

        # Optional GPU housekeeping (kept from original; can remove if undesired)
        try:
            import os, gc
            gc.collect()
            torch.cuda.empty_cache()
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        except Exception:
            pass
        
        # Apply Rotary Positional Embedding to q,k if provided
        if rotary_emb is not None:
            # rotary_emb should be callable: rotary_emb(seq_len) -> (cos, sin) with shape [seq_len, head_dim]
            cos, sin = rotary_emb(hw)  # cos,sin: [N, head_dim]
            # make broadcastable to [B, num_heads, N, head_dim]
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,N,D]
            sin = sin.unsqueeze(0).unsqueeze(0)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_score = (q@k.transpose(-1,-2)) * self.scale #attn_score[b, n_head, hw, hw]
        attn_score = attn_score.softmax(-1) # 在最后一个维度，即每一行进行softmax处理
        attn_score = self.attn_drop(attn_score) # softmax处理后要经过一个dropout层

        x = (attn_score@v).transpose(1,2).flatten(2) # v[b, n_head, hw, embed_dim_per_head]->x[b, hw, in_dim], in_dim=total_embed_dim=n_head*embed_dim_per_head
        x = self.proj(x)
        x = self.proj_drop(x) # 一般全连接后面都跟一个dropout层
        return x # x[b, hw, in_dim]
    

class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim=None, out_dim=None, act_func=nn.GELU, drop_ratio=0.):

        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim or in_dim
        self.out_dim = out_dim or in_dim
        

        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.act_func = act_func()
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)
        self.dropout = nn.Dropout(drop_ratio)

    
    def forward(self, x):
        # x[b, hw, in_dim=768]
        x = self.fc1(x)
        x = self.act_func(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x #x[b, hw, in_dim=768]


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 添加这个类：RoPE旋转位置编码支持
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rotary_pos_emb(q, k, cos, sin):
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q_rot = torch.stack([-q2, q1], dim=-1).reshape_as(q)
    k_rot = torch.stack([-k2, k1], dim=-1).reshape_as(k)
    q = q * cos + q_rot * sin
    k = k * cos + k_rot * sin
    return q, k









class EncoderBlock(nn.Module):
    '''该模块将重复12次'''

    def __init__(self, in_dim, num_heads, qk_scale, qkv_bias, attn_drop_ratio, drop_path_prob, drop_ratio=0., mlp_ratio=4., norm=nn.LayerNorm, act_func=nn.GELU):   # 默认参数必须在非默认参数之前！
        super().__init__()

        hidden_dim = int(mlp_ratio * in_dim)
        
        self.attn = Attention(in_dim, num_heads=num_heads, qk_scale=qk_scale, qkv_bias=qkv_bias, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio) #实例化
        self.drop = DropPath(drop_prob=drop_path_prob)  #也可以self.drop_path = DropPath(drop_prob) if drop_prob > 0. else nn.Identity()
        self.norm1 = norm(in_dim)
        self.norm2 = norm(in_dim)
        self.mlp = MLP(in_dim=in_dim, hidden_dim=hidden_dim, act_func=act_func, drop_ratio=drop_ratio) # out_dim直接用默认值，就不传入了。act_func=nn.GELU传入是为了可以在这一层修改，out_dim则一般不修改，就不传入


    def forward(self, x, rotary_emb=None):
        #x[b, hw, in_dim=768]
        x = x + self.drop(self.attn(self.norm1(x), rotary_emb=rotary_emb))
        x = x + self.drop(self.mlp(self.norm2(x))) 
        return x #x[b, hw, in_dim=768]



class VisionTransformer(nn.Module):

    def __init__(self, img_size=640, patch_size=16, in_c=3, embed_dim=768, depth=12, num_heads=12, 
                 norm_layer=None, qk_scale=None, qkv_bias=True, drop_ratio=0., attn_drop_ratio=0., drop_path_prob=0., mlp_ratio=4., 
                 act_func=None, embed_layer=PatchEmbed, representation_size=None, distilled=False):

        # representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set。对应的是最后的MLP中的pre-logits中的全连接层的节点个数。默认是none，也就是不会去构建MLP中的pre-logits，mlp中只有一个全连接层。


                 
        super().__init__()
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches #定义在该对象类（PatchEmbed）的内部
        self.grid_size = self.patch_embed.grid_size #(grid_size_H, grid_size_W)

         
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) ##norm_layer默认为none，所有norm_layer=nn.LayerNorm，用partial方法给一个默认参数。partial 函数的功能就是：把一个函数的某些参数给固定住，返回一个新的函数。
        act_func = act_func or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_prob, depth)] # 根据传入的drop_path_prob构建一个等差序列，总共depth个元素，即在每个Encoder Block中的失活比例都不一样。默认为0，可以传入参数改变。例如：depth=4, drop_path_ratio=0.1 → [0.0000, 0.0333, 0.0667, 0.1000]
        self.encoder_block = nn.Sequential(*[EncoderBlock(in_dim=embed_dim, num_heads=num_heads, qk_scale=qk_scale, qkv_bias=qkv_bias, 
                                          attn_drop_ratio=attn_drop_ratio, drop_path_prob=dpr[i], drop_ratio=drop_ratio, mlp_ratio=mlp_ratio,
                                            norm=norm_layer, act_func=act_func) for i in range(depth)])
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) #[1, HW, 768]
        
        # Rotary embedding: per-head dimension
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.rotary_emb = RotaryEmbedding(head_dim)
        
        #self.pos_drop = nn.Dropout(p=drop_ratio)
        self.norm = norm_layer(embed_dim)
        self.rotary_emb = RotaryEmbedding(embed_dim // num_heads)

        # distilled不用管，只用看representation_size即可，如果有传入representation_size，在MLP中就会构建pre-logits。否者直接 self.has_logits = False，然后执行self.pre_logits = nn.Identity()，相当于没有pre-logits。
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

    



    # # def forward_features(self, x):
    # def forward(self, x):
    #     # x[B, C, H, W]
    #     x0 = x[:,:3]
    #     x1 = x[:,3:]

    #     x0 = self.patch_embed(x0)  # x0[b, hw, 768]
    #     x0 = self.pos_drop(x0 + self.pos_embed) # x0[b, hw, 768]
    #     x1 = self.patch_embed(x1)  # x1[b, hw, 768]
    #     x1 = self.pos_drop(x1 + self.pos_embed) # x1[b, hw, 768]

    #     x = torch.cat([x0,x1], dim=1)
    #     x = self.encoder_block(x) # x[b, hw*2, 768]
    #     x = self.norm(x) # x[b, hw*2, 768]
    #     return x # x[b, hw*2, 768]

    def forward(self, x):
        x0 = x[:,:3]
        x1 = x[:,3:]

        x0 = self.patch_embed(x0)
        x1 = self.patch_embed(x1)

        x = torch.cat([x0, x1], dim=1)  # [B, 2HW, C]
        for blk in self.encoder_block:
            x = blk(x, rotary_emb=self.rotary_emb)
        x = self.norm(x)
        return x





    # def forward(self, x):
    #     # x[B, C, H, W]
        
    #     x = self.forward_features(x) #x [b, hw, 768]; 注意类内方法调用要加self.    

class SplitViT(nn.Module):
    def __init__(self, img_size=640, in_c=6, embed_dim=768, patch_size=16, depth=12, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.VIT1 = VisionTransformer(img_size=img_size, patch_size=patch_size, in_c=int(in_c/2), embed_dim=embed_dim, depth=depth, num_heads=num_heads, 
                 norm_layer=None, qk_scale=None, qkv_bias=True, drop_ratio=0., attn_drop_ratio=0., drop_path_prob=0., mlp_ratio=4., 
                 act_func=None, embed_layer=PatchEmbed, representation_size=None, distilled=False)
        # self.VIT2 = VisionTransformer(img_size=640, patch_size=patch_size, in_c=int(in_c/2), embed_dim=embed_dim, depth=depth, num_heads=num_heads, 
        #          norm_layer=None, qk_scale=None, qkv_bias=True, drop_ratio=0., attn_drop_ratio=0., drop_path_prob=0., mlp_ratio=4., 
                #  act_func=None, embed_layer=PatchEmbed, representation_size=None, distilled=False)
        self.grid_size = self.VIT1.patch_embed.grid_size #(grid_size_H, grid_size_W)
        

    def forward(self, x):
        # x[bs, 6, 640, 640]
        B = x.shape[0]
        # x0 = x[:,:3]
        # x1 = x[:,3:]
        x_out = self.VIT1(x).transpose(-1,-2) #x_out[b,768,hw*2]  # .reshape(B, self.embed_dim, self.grid_size[0], self.grid_size[1]) # x0_out[b, hw, 768]->x0_out[b, 768, h, w] 如[b, 768, 40, 40]
        #x1_out = self.VIT1(x1).transpose(-1,-2).reshape(B, self.embed_dim, self.grid_size[0], self.grid_size[1])
        
        #y = torch.cat([x0_out, x1_out], 1) # y[b, 768*2, h, w]
        return x_out #[b,768,hw*2]【最新代码hw*2==800】    #y





class MLPHead(nn.Module):
    """
    简单的 MLP 回归头，用于把 ViT 的 cls token 特征映射到 6 个仿射参数。
    可选 identity 初始化，保证初始预测接近恒等变换。
    """

    def __init__(self, embed_dim=768, out_dim=6, hidden_dim=None, init_identity=True, delta_scale=0.3, max_trans=0.5):  # delta_scale=0.05, max_trans=0.2
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        if hidden_dim==None:
            hidden_dim = embed_dim // 2
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.delta_scale = delta_scale
        self.max_trans = max_trans

        if init_identity:
            # 初始化为单位仿射变换 [1, 0, 0, 0, 1, 0]
            with torch.no_grad():
                self.fc2.weight.zero_()
                self.fc2.bias.copy_(torch.tensor([1, 1.4, 0, 0.5, 1, 0], dtype=torch.float))

    def forward(self, x):
        # x: [b, 1536, h, w]
        # B, C, H, W = x.shape
        # 需要先增加一个维度，变成4D张量
        x = x.unsqueeze(-1)  # 在最后增加一个维度 [16, 768, 800, 1] 
        x = self.pool(x).flatten(1)  #pool->[16,768,1,1] -> flatten(1) -> [16,768,1]  #原 -> [b, 1536]   
        
        
        x = F.relu(self.fc1(x))
        
        raw = self.fc2(x)  # [b, 6]
        # small deltas around identity

        # # 使用 tanh 限制范围
        # a00 = 1.0 + self.delta_scale * torch.tanh(raw[:,0])
        # a01 =      self.delta_scale * torch.tanh(raw[:,1])
        # tx   = self.max_trans * torch.tanh(raw[:,2])  # normalized translation
        # a10 =      self.delta_scale * torch.tanh(raw[:,3])
        # a11 = 1.0 + self.delta_scale * torch.tanh(raw[:,4])
        # ty   = self.max_trans * torch.tanh(raw[:,5])

        # theta = torch.stack([
        #     torch.stack([a00, a01, tx], dim=1),
        #     torch.stack([a10, a11, ty], dim=1)
        # ], dim=1)  # [B,2,3]

        angle = 1.0 * torch.tanh(raw[:,2])  # 第3个元素做旋转
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        a00 = (1 + self.delta_scale * torch.tanh(raw[:,0])) * cos
        a01 = -(1 + self.delta_scale * torch.tanh(raw[:,1])) * sin
        a10 = (1 + self.delta_scale * torch.tanh(raw[:,0])) * sin
        a11 = (1 + self.delta_scale * torch.tanh(raw[:,1])) * cos
        tx = self.max_trans * torch.tanh(raw[:,3])
        ty = self.max_trans * torch.tanh(raw[:,4])
        theta = torch.stack([torch.stack([a00, a01, tx],1), torch.stack([a10, a11, ty],1)],1)


        if self.training:
            theta.register_hook(lambda grad: torch.clamp(grad, -1, 1))


        # sx = torch.exp(raw[:,0])        # positive scale
        # sy = torch.exp(raw[:,1])
        # ang = raw[:,2]                  # rad, no tanh if you want full range
        # tx_px = raw[:,3]
        # ty_px = raw[:,4]

        # # convert pixel to normalized:
        # tx = tx_px / (W/2.0)
        # ty = ty_px / (H/2.0)

        # cos = torch.cos(ang); sin = torch.sin(ang)
        # a00 = sx * cos
        # a01 = -sin * sy
        # a10 = sin * sx
        # a11 = sy * cos
        # theta = torch.stack([torch.stack([a00,a01,tx],1), torch.stack([a10,a11,ty],1)], 1)

        return theta



class SpatialAffineSampler(nn.Module):
    def __init__(self, apply_to='imgA', mode='bilinear', padding='zeros'):
        super().__init__()
        self.apply_to = apply_to
        self.mode = mode
        self.padding = padding

    def forward(self, images, theta, i):
        """
        images: [b, 6, h, w]
        theta: [b, 6] -> [b, 2, 3]
        """
        B, C, H, W = images.shape
        # with torch.no_grad():
        #     print("theta mean:", theta.mean(dim=0))
        #     print("theta first sample:", theta[0])

        #theta = theta.view(-1, 2, 3)

        # debug：替换 theta 为 identity
        # theta = torch.zeros_like(theta)  # 先清零
        # theta[:, 0, 0] = 1.0  # x 轴缩放=1
        # theta[:, 1, 1] = 1.0  # y 轴缩放=1


        # theta_full = torch.zeros(theta.size(0), 3, 3, device=theta.device)
        # theta_full[:, :2, :] = theta
        # theta_full[:, 2, 2] = 1
        # theta_inv = torch.inverse(theta_full)[:, :2, :]  # 回到2×3
        # grid = F.affine_grid(theta_inv, size=(B, 3, H, W), align_corners=False)

        # 生成 grid
        grid = F.affine_grid(theta, size=(B, 3, H, W), align_corners=False)

        imgA = images[:, :3]  # 第一张图
        imgB = images[:, 3:]  # 第二张图

        imgA_warped = F.grid_sample(imgA, grid, mode=self.mode, padding_mode=self.padding, align_corners=False)
        
        import torchvision.utils as vutils
        if i%5==0:
            vutils.save_image(torch.cat([imgA, imgA_warped, imgB], dim=0), 
                      f"debug_step_{i}.png", nrow=B, normalize=True)
        
        out = torch.cat([imgA_warped, imgB], dim=1)  # 拼回 6 通道, [b,6,h,w]
        
        return out,theta

