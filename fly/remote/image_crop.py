import cv2
import math
import numpy as np
import random
import matplotlib.pyplot as plt
try:
    from fly.math2.transform import A232rot
except:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from fly.math2.transform import A232rot
    from general.config import load_config
import tifffile as tiff

   

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
    
def adjust_contrast_brightness(image, contrast, brightness):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

def random_crop_rotate(big_image, crop_size, aug_s=1.2, dbg=0):
    height, width, channels = big_image.shape
    s = random.uniform(1.0/aug_s, aug_s)
    angle = random.uniform(-math.pi, math.pi)
    w, h = crop_size
    center =  (random.uniform(w/2, width-w/2), random.uniform(h/2, height-h/2)) #center[2]
    xc,yc = center
    # 计算旋转矩阵
    #A23 = cv2.getRotationMatrix2D(center, angle, s) #A23[2,3]
    cos_t,sin_t = math.cos(angle),math.sin(angle)
    A23 = np.array([[ s*cos_t, s*sin_t, w/2 - s*( xc*cos_t + yc*sin_t)],
                    [-s*sin_t, s*cos_t, h/2 - s*(-xc*sin_t + yc*cos_t)]])
    
    if dbg:
        cos_t2,sin_t2, s2, xc2,yc2 = A232rot(A23,crop_size)
        assert(np.abs(cos_t2-cos_t)<1e-4 and np.abs(sin_t2-sin_t)<1e-4)
        assert(np.abs(s2-s)<1e-4)
        assert(np.abs(xc2-xc)<1e-4 and np.abs(yc2-yc)<1e-4)
    
    # 执行旋转
    rotated_image = cv2.warpAffine(big_image, A23, (w, h),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated_image.astype(np.float32)/255.0, A23,(xc/width,yc/height,s*cos_t,s*sin_t)

def get_crop_rotate(big_image, A23, crop_size=(640, 640)):
    rotated_image = cv2.warpAffine(big_image, A23, crop_size,
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated_image

def load_A23_file(path):
    with open(path, 'r') as fr:
        A23 = np.array([x.split() for x in fr.read().strip().splitlines() if len(x)], dtype=np.float32).reshape(-1, 2, 3)
    return A23

if __name__ == '__main__':
    config_path = 'fly/config-DOTA-remote.json'
    config = load_config(config_path)
    dir_path = config["images_dir"]#r'F:\代码及文档\remote\DOTA'

    crop_num = 15 #config.get('fly_num', 4)

    crop_size=(640, 640)

    from pathlib import Path
    from tqdm import tqdm
    dir_path = Path(dir_path)
    if dir_path.exists():
        IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
        img_files = [f for f in Path(dir_path).glob('*.*') if f.suffix[1:] in IMG_FORMATS]
        if len(img_files) > 0:
            crop_dir = dir_path.parent / 'crop_txt'
            crop_dir.mkdir(exist_ok=True)
            for f in tqdm(img_files, desc='Gen txt now...'):
                A23 = []
                if f.suffix == '.tif':
                    big_img = read_tif_with_tifffile(f)
                else:
                    big_img = cv2.imdecode(np.fromfile(f, dtype=np.uint8),cv2.IMREAD_COLOR) #aimage[H,W,C]
                for i in range(crop_num):
                    A23.append(random_crop_rotate(big_image=big_img,
                                                  crop_size=crop_size)[1].reshape(-1)
                            )
                with open(crop_dir / f.with_suffix('.txt').name, 'w') as fw:
                    fw.write('\n'.join([' '.join(['%.6f'] * 6) % (*a23,) for a23 in A23]))






