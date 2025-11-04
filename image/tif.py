import tifffile as tiff
import cv2
import numpy as np
from PIL import Image

def read_tif_with_tifffile(tif_path):
    try:
        image = tiff.imread(tif_path)
        if image is None:
            raise FileNotFoundError("图像未找到或加载失败。")
        # 归一化数组到范围 [0, 255]
        norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        norm_image = norm_image.astype(np.uint8)
        # 转换为 HxWxC 格式（高度，宽度，通道）
        if norm_image.shape[0] == 3:  # 检查第一个维度是否为通道
            norm_image = np.transpose(norm_image, (1, 2, 0))
        return norm_image
    except Exception as e:
        print(f"读取 TIFF 文件时出错：{e}")
        return None
    
def save_image_with_pillow(image, path):
    try:
        im = Image.fromarray(image)
        im.save(path)
        return True
    except Exception as e:
        print(f"使用 Pillow 保存图像时出错：{e}")
        return False