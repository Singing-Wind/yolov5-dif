import numpy as np

def np2image(image):
    image_vis = np.transpose(image, (1, 2, 0))
    # 如果是 float 类型（例如范围 [0, 1]），需要转为 uint8
    if image_vis.dtype == np.float32 or image_vis.dtype == np.float64:
        image_vis = (image_vis * 255).clip(0, 255).astype(np.uint8)
    return image_vis