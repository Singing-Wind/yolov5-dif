import os
import shutil
import json
import cv2
import numpy as np
from tqdm import tqdm

# iSAID 16类（顺序固定）
ISAID_CLASSES = ['Large_Vehicle','Small_Vehicle','plane', 'ship', 'tennis_court','Ground_Track_Field',
                 'Soccer_ball_field','Swimming_pool','baseball_diamond', 'storage_tank', 
                 'Helicopter','basketball_court', 'Roundabout', 'Bridge', 'Harbor'
]

# COCO category_id (1-based) 映射到 YOLO id (0-based)
CATEGORY_ID_TO_YOLO = {i+1: i for i in range(len(ISAID_CLASSES))}

def compute_coefficients_interp(xy, terms=12, interp=True, return_list=False):
    x = np.array(xy[0::2])
    y = np.array(xy[1::2])
    if interp:
        x = np.concatenate([x, [x[0]]], dtype=np.float32)
        y = np.concatenate([y, [y[0]]], dtype=np.float32)
        ori = np.linspace(0, 1, x.shape[0], endpoint=True)
        gap = np.linspace(0, 1, terms * 2, endpoint=False)
        x = np.interp(gap, ori, x)
        y = np.interp(gap, ori, y)
    N = len(x)
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    a0 = 1./N * np.sum(x)
    c0 = 1./N * np.sum(y)
    an = np.zeros(terms+1)
    bn = np.zeros(terms+1)
    cn = np.zeros(terms+1)
    dn = np.zeros(terms+1)
    for k in range(1, terms+1):
        an[k] = 2./N * np.sum(x * np.cos(k*t))
        bn[k] = 2./N * np.sum(x * np.sin(k*t))
        cn[k] = 2./N * np.sum(y * np.cos(k*t))
        dn[k] = 2./N * np.sum(y * np.sin(k*t))
    if return_list:
        coef_list = [a0, c0]
        for k in range(1, len(an)):
            coef_list.extend([an[k], bn[k], cn[k], dn[k]])
        return coef_list
    an[0] = a0
    cn[0] = c0
    return (an, bn, cn, dn)

def polygon_to_yolo(polygon, img_w, img_h):
    x = polygon[:,0]
    y = polygon[:,1]
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height

def process_image(image_path, mask_path, coco_annotation, output_image_dir, output_label_dir, ft_terms=12):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 读取图片失败: {image_path}")
        return
    h, w = image.shape[:2]
    img_filename = os.path.basename(image_path)

    mask_rgb = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask_rgb is None:
        print(f"❌ 读取Instance_mask失败: {mask_path}")
        return

    # iSAID的Instance_id_RGB是RGB编码，OpenCV读进来是BGR，需要转换顺序
    r = mask_rgb[:, :, 2].astype(np.uint32)
    g = mask_rgb[:, :, 1].astype(np.uint32)
    b = mask_rgb[:, :, 0].astype(np.uint32)
    mask_ids = r + (g << 8) + (b << 16)

    # 找到image_id
    image_id = None
    for img_info in coco_annotation['images']:
        if img_info['file_name'] == img_filename:
            image_id = img_info['id']
            break
    if image_id is None:
        print(f"⚠️ 图片ID未找到: {img_filename}")
        return

    image_annos = [ann for ann in coco_annotation['annotations'] if ann['image_id'] == image_id]

    yolo_lines = []
    ft_lines = []

    for ann in image_annos:
        category_id = ann['category_id']
        if category_id not in CATEGORY_ID_TO_YOLO:
            continue
        cls_id = CATEGORY_ID_TO_YOLO[category_id]

        # 取instance_id，优先用注释的instance_id字段，否则用id
        instance_id = ann.get('instance_id', ann['id'])
        if instance_id == 0:
            # 0 是背景，跳过
            continue

        mask_instance = (mask_ids == instance_id)
        if not np.any(mask_instance):
            print(f"⚠️ mask中找不到实例id {instance_id}，跳过")
            continue

        mask_uint8 = (mask_instance.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            print(f"⚠️ 实例id {instance_id} 无轮廓，跳过")
            continue

        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 6:
            print(f"⚠️ 实例id {instance_id} 轮廓点太少({len(contour)})，跳过傅里叶")
            continue

        contour_pts = contour[:, 0, :]  # (N,2)
        xy_flat = []
        for pt in contour_pts:
            xy_flat.append(pt[0] / w)
            xy_flat.append(pt[1] / h)

        ft_coeffs = compute_coefficients_interp(xy_flat, terms=ft_terms, return_list=True)
        bbox = polygon_to_yolo(contour_pts, w, h)

        yolo_lines.append(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
        ft_lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in ft_coeffs))

    if yolo_lines:
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        dst_img_path = os.path.join(output_image_dir, img_filename)
        if not os.path.exists(dst_img_path):
            shutil.copyfile(image_path, dst_img_path)

        label_txt_path = os.path.join(output_label_dir, img_filename.replace('.png', '.txt'))
        label_ft_path = os.path.join(output_label_dir, img_filename.replace('.png', '.ft'))
        with open(label_txt_path, 'w') as ftxt, open(label_ft_path, 'w') as fft:
            ftxt.write("\n".join(yolo_lines))
            fft.write("\n".join(ft_lines))

def batch_process_isaid(root_dir, split='train', output_dir='output', ft_terms=12):
    images_dir = os.path.join(root_dir, split, 'images')
    masks_dir = os.path.join(root_dir, split, 'Instance_masks', 'images')
    annotations_file = os.path.join(root_dir, split, 'Annotations', f'iSAID_{split}_2019.json')

    with open(annotations_file, 'r') as f:
        coco_ann = json.load(f)

    img_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    img_files.sort()

    out_img_dir = os.path.join(output_dir, split, 'images')
    out_label_dir = os.path.join(output_dir, split, 'labels')

    for img_file in tqdm(img_files, desc=f"Processing {split}"):
        img_path = os.path.join(images_dir, img_file)
        mask_file = img_file.replace('.png', '_instance_id_RGB.png')
        mask_path = os.path.join(masks_dir, mask_file)
        if not os.path.exists(mask_path):
            print(f"⚠️ 找不到掩码文件: {mask_path}")
            continue
        print(f"正在处理图片: {img_path}, 掩码: {mask_path}")
        process_image(img_path, mask_path, coco_ann, out_img_dir, out_label_dir, ft_terms=ft_terms)

if __name__ == "__main__":
    root_isaid_dir = ""  # 这里改成你的iSAID数据集根目录
    output_root = "iSAID_yolo_mask_output"

    batch_process_isaid(root_isaid_dir, split='train', output_dir=output_root)
    batch_process_isaid(root_isaid_dir, split='val', output_dir=output_root)
