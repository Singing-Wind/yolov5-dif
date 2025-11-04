
import numpy as np

def normalize_dim(x, dim=0, eps=1e-8):
    #在指定维度上进行L2单位化
    norm = x.norm(p=2, dim=dim, keepdim=True)  # 计算L2范数
    return x / (norm + eps)  # 避免除以零

def convert_image(img):
    # 将 [H, W, C] uint8 图像转为 [C, H, W] float32 并归一化到 [0, 1]
    assert img.dtype == np.uint8, "输入必须是 uint8 类型图像"
    img = img.astype(np.float32) / 255.0  # 转为 float32 并归一化
    img = np.transpose(img, (2, 0, 1))    # 从 [H, W, C] 转为 [C, H, W]
    return img

def recover_image(tensor):
    # 将 [C, H, W] float32 归一化图像还原为 [H, W, C] uint8 图像
    # 输入：
    #     tensor: numpy.ndarray, shape=(C, H, W), dtype=float32，值在 [0, 1]
    # 返回：
    #     img: numpy.ndarray, shape=(H, W, C), dtype=uint8
    assert tensor.ndim == 3 and tensor.shape[0] in [1, 3], "输入必须是 [C, H, W] 格式"
    img = np.transpose(tensor, (1, 2, 0))           # [C, H, W] → [H, W, C]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)  # 反归一化并转为 uint8
    return img
