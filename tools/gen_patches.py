import os
import shutil
import cv2
import numpy as np
import random
from tqdm import tqdm

import torch
torch.manual_seed(88)
import random
random.seed(88)
import numpy as np
np.random.seed(88)

# 全局变量
scale = 1.0 #scale>1时放大，app scale = master_obj_scale / app_obj_size, master scale=1.0
PATCH_SIZE = (640, 640)  # [宽, 高]
MAX_OBJ_SELECT = 3
MAX_NEG_SELECT = 3
MIN_DISTANCE_TO_EDGE = 30

def get_affine_transform(center, s, angle, dst_offset):
    angle_rad = np.deg2rad(angle)
    cos_rad = np.cos(angle_rad)
    sin_rad = np.sin(angle_rad)
    cx,cy = center
    dx,dy = dst_offset
    transform_matrix = np.array([
        [ s*cos_rad, s*sin_rad, dx - s*( cos_rad*cx+sin_rad*cy)],
        [-s*sin_rad, s*cos_rad, dy - s*(-sin_rad*cx+cos_rad*cy)]
    ])
    return transform_matrix

def read_image_label(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None

    base_name, _ = os.path.splitext(os.path.basename(image_path))
    base_dir = os.path.dirname(image_path)
    txt_path = os.path.join(base_dir.replace('images', 'labels'), f"{base_name}.txt")
    pts_path = os.path.join(base_dir.replace('images', 'labels'), f"{base_name}.pts")

    with open(txt_path, "r") as f:
        txt_data = [list(map(float, line.strip().split())) for line in f.readlines() if len(line.strip().split()) == 5]

    with open(pts_path, "r") as f:
        pts_data = [list(map(float, line.strip().split())) for line in f.readlines() if len(line.strip().split()) == 8]

    assert len(txt_data) == len(pts_data)

    return image, txt_data, pts_data

def save_patch(image, label_txt, label_pts, patch_name, output_path):
    assert len(label_txt) == len(label_pts)

    image_name = f"{output_path}/images/{patch_name}.jpg"
    txt_name = f"{output_path}/labels/{patch_name}.txt"
    pts_name = f"{output_path}/labels/{patch_name}.pts"

    cv2.imwrite(image_name, image)

    if len(label_txt) > 0:
        with open(txt_name, "w") as f:
            for line in label_txt:
                f.write(f"{int(line[0])} " + " ".join(map(str, line[1:])) + "\n")

        with open(pts_name, "w") as f:
            for line in label_pts:
                f.write(" ".join(map(str, line)) + "\n")

def generate_patches(data_path, diz, diz_angle, diz_scale, min_dist_to_edge, max_obj_select, max_neg_select,output_folder='patches-py',max_generated=0):
    image_dir = os.path.join(data_path, 'images')
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'))]

    output_path = os.path.join(data_path, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # 拷贝 names.txt 文件到新数据集路径
    names_file = os.path.join(data_path, 'names.txt')
    if os.path.exists(names_file):
        shutil.copy(names_file, output_path)
    else:
        print(f"Warning: {names_file} does not exist.")

    if not os.path.exists(os.path.join(output_path, 'images')):
        os.makedirs(os.path.join(output_path, 'images'))
    if not os.path.exists(os.path.join(output_path, 'labels')):
        os.makedirs(os.path.join(output_path, 'labels'))

    total_img = 0
    for image_path in tqdm(image_files, total=len(image_files)):
        image, txt_data, pts_data = read_image_label(image_path)
        if image is None or txt_data is None or pts_data is None:
            print(f"Failed to read {image_path}")
            continue

        height, width = image.shape[:2]
        select_times = [0] * len(txt_data)

        filename_with_extension = os.path.basename(image_path)
        filename, _ = os.path.splitext(filename_with_extension)

        #gen positive samples
        patch_index = 0
        while True:
            subset = [i for i, t in enumerate(select_times) if t < max_obj_select]
            if not subset:
                break

            idx = random.choice(subset)
            x_center, y_center = txt_data[idx][1] * width, txt_data[idx][2] * height
            dx, dy = random.randint(-diz, diz), random.randint(-diz, diz)
            #angle = random.choice([-180, -90, 0, 90])
            angle = random.uniform(-diz_angle,diz_angle)

            #matrix = cv2.getRotationMatrix2D((x_center + dx, y_center + dy), angle, 1.0)
            matrix = get_affine_transform((x_center + dx, y_center + dy),scale * random.uniform(1.0/diz_scale, diz_scale),angle,(PATCH_SIZE[0]/2,PATCH_SIZE[1]/2))
            patch = cv2.warpAffine(image, matrix, PATCH_SIZE)
            # cv2.imshow('patch',patch)
            # cv2.waitKey(0)

            new_txt_data, new_pts_data = [], []
            for j, pts in enumerate(pts_data):
                pts = np.array(pts).reshape(4, 2) * [width, height]
                new_pts = cv2.transform(np.array([pts]), matrix)[0]
                new_pts = new_pts / PATCH_SIZE  # 将绝对坐标转换为相对坐标
                # 计算新坐标的中心点
                center_x = np.mean(new_pts[:, 0])
                center_y = np.mean(new_pts[:, 1])
                #if all(0 <= p[0] < 1 and 0 <= p[1] < 1 for p in new_pts):
                if 0 <= center_x < 1 and 0 <= center_y < 1:
                    x_min, y_min = new_pts[:, 0].min(), new_pts[:, 1].min()
                    x_max, y_max = new_pts[:, 0].max(), new_pts[:, 1].max()
                    new_x = (x_min + x_max) / 2
                    new_y = (y_min + y_max) / 2
                    new_w = (x_max - x_min)
                    new_h = (y_max - y_min)
                    new_txt_data.append([txt_data[j][0], new_x, new_y, new_w, new_h])
                    new_pts_data.append(new_pts.flatten().tolist())

                    if min(x_min*PATCH_SIZE[0], (1 - x_max)*PATCH_SIZE[0], y_min*PATCH_SIZE[1], (1 - y_max)*PATCH_SIZE[1]) > min_dist_to_edge:
                        select_times[j] += 1

            patch_name = f'{filename}_{idx}-{dx}_{dy}-{angle}-{patch_index}'
            save_patch(patch, new_txt_data, new_pts_data, patch_name, output_path)
            patch_index += 1
            
        #gen negitive samples
        for _ in range(max_neg_select):
            x_center, y_center = random.randint(min_dist_to_edge, width - min_dist_to_edge), random.randint(min_dist_to_edge, height - min_dist_to_edge)
            angle = random.uniform(-diz_angle,diz_angle)
            #matrix = cv2.getRotationMatrix2D((x_center, y_center), angle, scale)
            matrix = get_affine_transform((x_center, y_center), scale * random.uniform(1.0/diz_scale, diz_scale),angle,(PATCH_SIZE[0]/2,PATCH_SIZE[1]/2))
            patch = cv2.warpAffine(image, matrix, PATCH_SIZE)

            for j, pts in enumerate(pts_data):
                pts = np.array(pts).reshape(4, 2) * [width, height]
                new_pts = cv2.transform(np.array([pts]), matrix)[0]
                new_pts = new_pts / PATCH_SIZE  # 将绝对坐标转换为相对坐标

                center_x = np.mean(new_pts[:, 0])
                center_y = np.mean(new_pts[:, 1])
                #if all(0 <= p[0] < 1 and 0 <= p[1] < 1 for p in new_pts):
                if 0 <= center_x < 1 and 0 <= center_y < 1:
                    cv2.fillPoly(patch, [np.int32(new_pts * PATCH_SIZE)], (128, 128, 128))

            patch_name = f'neg-{os.path.basename(image_path)}_{x_center}_{y_center}-{patch_index}'
            save_patch(patch, [], [], patch_name, output_path)
            patch_index += 1
        total_img+=1
        if max_generated>0 and total_img>max_generated:
            break

if __name__ == '__main__':
    # 示例用法
    # data_path = '/media/data4T/datas/SAR-TZ-ship'
    #data_path ='/media/liu/088d8f6e-fca3-4aed-871f-243ad962413b/datas/car_det_train/val'
    #data_path = '/media/liu/088d8f6e-fca3-4aed-871f-243ad962413b/datas/dota1.5/generated_car'
    data_path = '/media/data4T/datas/SAR-AIR-SARShip'
    generate_patches(data_path, 36, 180, 2.0, MIN_DISTANCE_TO_EDGE, MAX_OBJ_SELECT, MAX_NEG_SELECT,output_folder='val-patches',max_generated=0)
