import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

def build_image(base_img, wh):
    M = np.float32([[1, 0, 0], [0, 1, 0]])
    warped = cv2.warpAffine(base_img, M, wh, borderMode=cv2.BORDER_REFLECT_101)
    return warped

def build_label_xywh(labels, wh_before, wh):
    mul_num = int(np.ceil(wh[0] / wh_before[0])), int(np.ceil(wh[1] / wh_before[1]))
    scale = [wh_before[i] / wh[i] for i in range(2)]
    labels_base = labels.copy()
    for i in range(1, mul_num[0]):  # x
        labels_exp1 = labels_base.copy()
        labels_exp1[:, 1] = 2 * ((i + 1) // 2) + labels_exp1[:, 1] * (1 if i % 2 == 0 else -1)
        labels = np.concatenate([labels, labels_exp1], axis=0)
    labels_base = labels.copy()
    for i in range(1, mul_num[1]):  # y
        labels_exp1 = labels_base.copy()
        labels_exp1[:, 2] = 2 * ((i + 1) // 2) + labels_exp1[:, 2] * (1 if i % 2 == 0 else -1)
        labels = np.concatenate([labels, labels_exp1], axis=0)
    labels = labels * np.asarray([[1, scale[0], scale[1], scale[0], scale[1]]])
    return labels


def build_label_pts(labels, wh_before, wh):
    mul_num = int(np.ceil(wh[0] / wh_before[0])), int(np.ceil(wh[1] / wh_before[1]))
    scale = [wh_before[i] / wh[i] for i in range(2)]
    labels_base = labels.copy()
    for i in range(1, mul_num[0]):
        labels_exp1 = labels_base.copy()
        labels_exp1[:, 0::2] = 2 * ((i + 1) // 2) + labels_exp1[:, 0::2] * (1 if i % 2 == 0 else -1)
        labels = np.concatenate([labels, labels_exp1], axis=0)
    labels_base = labels.copy()
    for i in range(1, mul_num[1]):
        labels_exp1 = labels_base.copy()
        labels_exp1[:, 1::2] = 2 * ((i + 1) // 2) + labels_exp1[:, 1::2] * (1 if i % 2 == 0 else -1)
        labels = np.concatenate([labels, labels_exp1], axis=0)
    labels = labels * np.asarray([[scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1]]])
    return labels

import shutil
def process_images(dir_path, wh, save_dir=None, save_labels=True):
    dir_path = Path(dir_path)
    labels_path = dir_path.parent / 'labels'
    images = [f for f in dir_path.rglob('*.*') if f.suffix[1:] in IMG_FORMATS]
    if len(images) > 0:
        if save_dir is None:
            save_dir = dir_path.parent / 'expend_data'
        save_dir = Path(save_dir)
        save_img = save_dir / 'images'
        save_lab = save_dir / 'labels'
        save_img.mkdir(parents=True, exist_ok=True)
        for image in tqdm(images, desc="Expending now..."):
            img = cv2.imdecode(np.fromfile(image, dtype=np.uint8),cv2.IMREAD_COLOR)
            cv2.imencode('.jpg', build_image(img, wh))[1].tofile(save_img / image.with_suffix('.jpg').name)
            if save_labels and labels_path.exists():
                wh_before = img.shape[1], img.shape[0]
                if not save_lab.exists():
                    save_lab.mkdir(parents=True)
                lb_file = labels_path / image.with_suffix('.txt').name
                if lb_file.exists():
                    with open(lb_file, 'r') as f:
                        labels = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        labels = np.array(labels, dtype=np.float32)
                    if labels.shape[0] > 0:
                        labels = build_label_xywh(labels, wh_before, wh) #labels[nt,5]
                        lp_file = lb_file.with_suffix('.pts')
                        mask = np.all(labels[:, 1:3] < 1, axis=-1)
                        labels = labels[mask]
                        string_out = '\n'.join([('%d' + ' %.6f' * 4) % (*label,) for label in labels])
                        with open(save_lab / lb_file.name, 'w') as f:
                            f.write(string_out)
                        if lp_file.exists():
                            with open(lp_file, 'r') as f:
                                labels_pts = [x.split() for x in f.read().strip().splitlines() if len(x)]
                                labels_pts = np.array(labels_pts, dtype=np.float32)
                            labels_pts = build_label_pts(labels_pts, wh_before, wh)[mask]
                            string_out = '\n'.join([('%.6f' + ' %.6f' * 7) % (*label,) for label in labels_pts])
                            with open(save_lab / lp_file.name, 'w') as f:
                                f.write(string_out)
        # 复制文件（文件名不变）
        shutil.copy2(str(dir_path.parent / 'names.txt'), save_dir)
                    


if __name__ == '__main__':
    # dir_path = r'/media/data4T/datas/SAR-mstar/images'
    # dir_path = r'E:\datas\coco128\images'
    dir_path = '/media/data4T/datas/SAR-SSDD SSDD+/val/images'
    save_dir = None     # None 则保存在 dir_path.parent / 'expend_data' 文件夹下
    wh = (4096, 4096)   # 生成的图像大小
    save_labels = True  # True 时使用  dir_path.parent / 'labels' 文件夹内的txt和pts文件 生成标签, 根据目标中心点是否在生成图内进行过滤
    process_images(dir_path, wh, save_dir, save_labels)
    print('Done.')
