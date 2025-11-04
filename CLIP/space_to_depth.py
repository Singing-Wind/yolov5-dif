import torch
import torch.nn.functional as F

def space_to_depth(image, block_size=4):
    # 将图像的 H, W 缩小为 1/block_size，通道扩展为 block_size^2 倍。
    # Args:
    #     image: Tensor [B, C, H, W]
    #     block_size: 缩放因子（4 表示缩小 4 倍）
    # Returns:
    #     Tensor [B, block_size^2, C, H // block_size, W // block_size]
    B, C, H, W = image.shape
    assert H % block_size == 0 and W % block_size == 0, "H and W must be divisible by block_size"

    # reshape and permute to bring spatial blocks into channels
    image = image.view(B, C, H // block_size, block_size, W // block_size, block_size)
    image = image.permute(0, 3, 5, 1, 2, 4).contiguous()
    image = image.view(B, block_size * block_size, C, H // block_size, W // block_size)
    return image

def depth_to_space(tensor, block_size=4):
    B, C, H, W = tensor.shape
    assert C % (block_size ** 2) == 0, "C must be divisible by block_size^2"

    tensor = tensor.view(B, C // (block_size ** 2), block_size, block_size, H, W)
    tensor = tensor.permute(0, 1, 4, 2, 5, 3).contiguous()
    tensor = tensor.view(B, C // (block_size ** 2), H * block_size, W * block_size)
    return tensor