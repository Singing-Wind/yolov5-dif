import shutil
import random
import os
from pathlib import Path
from tqdm import tqdm

# 源目录和目标目录
class DataSplit(object):
    SAMPLE_FORMAT = ('.jpg', '.jpeg','.bmp','.png', '.csv')
    LABEL_FORMAT = ('.txt', '.pts', '.ft', '.xyz', '.abc', '.dir')
    def __init__(self, allpath, save_path=None, ):
        allpath = Path(allpath).absolute()
        self.root = allpath
        self.images_dir = allpath / 'images'
        self.labels_dir = allpath / 'labels'
        self.val_dir = allpath / 'val' if save_path is None else Path(save_path)
        self.val_images_dir = self.val_dir / 'images'
        self.val_labels_dir = self.val_dir / 'labels'
        self.val_calib_dir = self.val_dir / 'calib'

    def __call__(self, sample_ratio=0.1):
        # 创建目标目录
        self.val_images_dir.mkdir(parents=True, exist_ok=True)
        self.val_labels_dir.mkdir(parents=True, exist_ok=True)
        self.val_calib_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(self.root / 'names.txt', self.val_dir / 'names.txt')
        if os.path.exists(str(self.root / 'Camera.txt')):
            shutil.copy(self.root / 'Camera.txt', self.val_dir / 'Camera.txt')

        # 获取 images 文件列表
        image_files = [f for f in self.images_dir.glob('*.*') if f.suffix.lower() in  self.SAMPLE_FORMAT]
        assert self.images_dir.parent==self.root
        calib_path = self.root / 'calib'

        # 取 ??% 样本
        num_samples = len(image_files)
        num_to_move = int(num_samples * sample_ratio)
        sample_indices = random.sample(range(num_samples), num_to_move)

        # 转移文件
        for idx in tqdm(sample_indices, desc="Moving..."):
            img_name = image_files[idx]
            # find calib
            calib_name = (os.path.basename(os.path.splitext(img_name)[0])+'.txt')
            calib_path_name = str(calib_path / calib_name)
            if os.path.exists(calib_path_name):
                shutil.move(calib_path_name, self.val_calib_dir / calib_name)
            # 移动图片
            shutil.move(img_name, self.val_images_dir / img_name.name)
            # 移动标签
            for suf in self.LABEL_FORMAT:
                lbl_name = self.labels_dir / img_name.with_suffix(suf).name
                if lbl_name.exists():
                    shutil.move(lbl_name, self.val_labels_dir / f"{img_name.stem}{lbl_name.suffix}")

        print(f"Moved {num_to_move} samples to {self.val_dir}.")

if __name__ == '__main__':
    # all_dir = r'H:\datas\nuScenes\v1.0-mini\yolo6cams-0'
    all_dir = '/media/data4T/datas/SAR-SSDD SSDD+'
    ds = DataSplit(all_dir, save_path=None)
    ds(sample_ratio=0.15)