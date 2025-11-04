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
globe_scale = 1.0 #global_scale>1时放大，app global_scale = master_obj_scale / app_obj_size, master global_scale=1.0
PATCH_SIZE = (640, 640)  # [宽, 高]
MAX_OBJ_SELECT = 3
MAX_NEG_SELECT = 3
MIN_DISTANCE_TO_EDGE = 8

import os
import numpy as np

def pol_scale(pol,scale):
    assert ((len(pol))%2) == 0
    scalex,scaley = scale
    pol[0::2] *= scalex  # a_i
    pol[1::2] *= scaley  # c_i

def process_pol(pol_line, A23, image_size, patch_size):
    # 输入：
    #     pol_line: list, shape=[4,2]，第一个为类别id，其余为傅里叶系数
    #     patch_size: (W, H)
    #     A23: 2x3仿射矩阵
    # 输出：
    #     yolo_label: YOLO格式字符串（class_id xc yc w h）
    #     pol_line: list，以 class_id 开头的归一化傅里叶系数
    Height,Width = image_size
    W, H = patch_size
    pol_line = np.array(pol_line, dtype=np.float32) #pol_line[1+2n]

    class_id = int(pol_line[0])
    poly = pol_line[1:].copy() #poly[2n]
    assert poly.shape[0]%2==0

    # 缩放傅里叶系数（归一化 → 像素）
    pol_scale(poly,(Width,Height))

    A = A23[:, :2]
    t = A23[:, 2:]

    pol2 = (A@poly.reshape(-1,2).T + t).T #pol2[n,2]

    xc2,yc2 = pol2.mean(0) #xc,yc

    if (0 <= xc2 <= W and 0 <= yc2 <= H):
        # 重建闭合曲线
        x,y = pol2.T #x[n],y[n]

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        xc = (xmin + xmax) / 2 / W
        yc = (ymin + ymax) / 2 / H
        w = (xmax - xmin) / W
        h = (ymax - ymin) / H

        yolo_label = [class_id, xc, yc, w, h]

        # 返回 class_id 开头的归一化系数
        ft_norm = pol2.flatten() #ft_norm[2n]
        pol_scale(ft_norm,(1.0/W,1.0/H))
        pol_line = [class_id] + ft_norm.tolist()

        #xy(x,y)->p[2,npts]->p[npts,2]
        pts = np.vstack(list((x,y))).T #p[2,npts]->p[npts,2]

        return yolo_label, pol_line, pts
    else:
        return None, None, None

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
        return None, None, None, None

    base_name, _ = os.path.splitext(os.path.basename(image_path))
    base_dir = os.path.dirname(image_path)
    txt_path = os.path.join(base_dir.replace('images', 'labels'), f"{base_name}.txt")
    pol_path = os.path.join(base_dir.replace('images', 'labels'), f"{base_name}.pol")
    ft_path = os.path.join(base_dir.replace('images', 'labels'), f"{base_name}.ft")

    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            txt_data = [list(map(float, line.strip().split())) for line in f.readlines() if len(line.strip().split()) == 5]
    else:
        txt_data = None
    
    if os.path.exists(pol_path):
        with open(pol_path, "r") as f:
            pol_data = [list(map(float, line.strip().split())) for line in f.readlines() if len(line.strip().split()) >= 4]
        assert txt_data == None or len(pol_data) == len(txt_data)
    else:
        pol_data == None

    return image, txt_data, pol_data

def save_patch(image, new_pol_data, label_txt, patch_name, output_path):
    assert label_txt==None or len(label_txt) == len(new_pol_data)

    image_name = f"{output_path}/images/{patch_name}.jpg"
    txt_name = f"{output_path}/labels/{patch_name}.txt"
    pol_name = f"{output_path}/labels/{patch_name}.pol"

    cv2.imwrite(image_name, image)

    if new_pol_data is not None and len(new_pol_data)>0 and new_pol_data[0]is not None:
        with open(pol_name, "w") as f:
            for line in new_pol_data:
                f.write(" ".join(map(str, line)) + "\n")

    if label_txt is not None and len(label_txt)>0 and label_txt[0]is not None:
        if len(label_txt) > 0:
            with open(txt_name, "w") as f:
                for line in label_txt:
                    f.write(f"{int(line[0])} " + " ".join(map(str, line[1:])) + "\n")

colors = [
        (54, 67, 244),
        # (99, 30, 233),
        (176, 39, 156),
        (183, 58, 103),
        (181, 81, 63),
        (243, 150, 33),
        (212, 188, 0),
        (136, 150, 0),
        (80, 175, 76),
        (74, 195, 139),
        (57, 220, 205),
        (59, 235, 255),
        (0, 152, 255),
        (34, 87, 255),
        (72, 85, 121),
        (180, 105, 255)]
def generate_patches(data_path, diz, diz_angle, diz_scale, min_dist_to_edge, npts, max_obj_select, max_neg_select,output_folder='patches',max_generated=0,draw_polys_on_big=0):
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
    total_patches = 0
    for image_path in tqdm(image_files, total=len(image_files)):
        image, txt_data, pol_data = read_image_label(image_path)
        if image is not None and (pol_data is not None or (txt_data is not None)):
            height, width = image.shape[:2]
            select_times = [0] * (len(pol_data) if pol_data is not None else len(txt_data))

            filename_with_extension = os.path.basename(image_path)
            filename, _ = os.path.splitext(filename_with_extension)

            #gen positive samples
            patch_index = 0
            while True:
                subset = [i for i, t in enumerate(select_times) if t < max_obj_select]
                if subset:
                    idx = random.choice(subset)
                    if pol_data is not None:
                        pol = pol_data[idx] #pol[1+2n]
                        x_center, y_center = np.mean(pol[1::2]) * width, np.mean(pol[2::2]) * height
                    elif txt_data is not None:
                        x_center, y_center = txt_data[idx][1] * width, txt_data[idx][2] * height
                    dx, dy = random.randint(-diz, diz), random.randint(-diz, diz)
                    #angle = random.choice([-180, -90, 0, 90])
                    angle = random.uniform(-diz_angle,diz_angle)

                    #A23 = cv2.getRotationMatrix2D((x_center + dx, y_center + dy), angle, 1.0)
                    A23 = get_affine_transform((x_center + dx, y_center + dy),globe_scale * random.uniform(1.0/diz_scale, diz_scale),angle,(PATCH_SIZE[0]/2,PATCH_SIZE[1]/2))
                    patch = cv2.warpAffine(image, A23, PATCH_SIZE)
                    # cv2.imshow('patch',patch)
                    # cv2.waitKey(0)

                    new_txt_data, new_pol_data = [], []
                    if pol_data is not None:
                        for j, pol_line in enumerate(pol_data):
                            yolo_label, pol_line, pts = process_pol(pol_line, A23, (height,width), PATCH_SIZE)
                            if pol_line is not None:
                                new_txt_data.append(yolo_label)
                                new_pol_data.append(pol_line)
                                xy = yolo_label[1:3]
                                xy[0]*=PATCH_SIZE[0]
                                xy[1]*=PATCH_SIZE[1]
                                if min(xy[0],PATCH_SIZE[0]-xy[0],xy[1],PATCH_SIZE[1]-xy[1]) >  min_dist_to_edge:
                                    select_times[j] += 1

                    patch_name = f'{filename}_{idx}-{dx}_{dy}-{angle}-{patch_index}'
                    save_patch(patch, new_pol_data, None, patch_name, output_path) #new_txt_data
                    patch_index += 1
                else:
                    break
                
            # gen negitive samples
            for _ in range(max_neg_select):
                x_center, y_center = random.randint(min_dist_to_edge, width - min_dist_to_edge), random.randint(min_dist_to_edge, height - min_dist_to_edge)
                angle = random.uniform(-diz_angle,diz_angle)
                #A23 = cv2.getRotationMatrix2D((x_center, y_center), angle, globe_scale)
                A23 = get_affine_transform((x_center, y_center), globe_scale * random.uniform(1.0/diz_scale, diz_scale),angle,(PATCH_SIZE[0]/2,PATCH_SIZE[1]/2))
                patch = cv2.warpAffine(image, A23, PATCH_SIZE)

                if pol_data is not None:
                    for j, pol_line in enumerate(pol_data):
                        yolo_label, pol_line, pts = process_pol(pol_line, A23, (height,width), PATCH_SIZE)
                        if yolo_label is not None:
                            # 转为 int32，并 reshape 成 OpenCV 需要的格式
                            pts = pts.reshape((-1, 1, 2)).astype(np.int32) #pts[npts,2]->pts[npts,1,2]
                            # cv2.polylines(patch,[pts],True,colors[id%len(colors)])
                            cv2.fillPoly(patch, [pts], (128, 128, 128))

                patch_name = f'neg-{os.path.basename(image_path)}_{x_center}_{y_center}-{patch_index}'
                save_patch(patch, [], [], patch_name, output_path)
                patch_index += 1

            if draw_polys_on_big > 0:
                if pol_data is not None:
                    for j, pol_line in enumerate(pol_data):
                        pol_line = np.array(pol_line, dtype=np.float32)

                        class_id = int(pol_line[0])
                        pol = pol_line[1:] #pol[2n]

                        pol = np.array(pol)#pol[2n]

                        # 缩放傅里叶系数（归一化 → 像素）
                        pol_scale(pol,(width,height))

                        pts = pol.reshape((-1, 1, 2)).astype(np.int32) #pol[npts,2]->pts[npts,1,2]
                        cv2.polylines(image,[pts],True,colors[class_id%len(colors)],thickness=2)

                image_name = f"{output_path}/{filename}.jpg"
                cv2.imwrite(image_name, image)
                draw_polys_on_big-=1

            total_img+=1
            total_patches+=patch_index
            if max_generated>0 and total_img>max_generated:
                break
        else:
            print(f"Failed to read {image_path}")
    return total_patches

if __name__ == "__main__":
    # 示例用法
    # data_path = '/media/data4T/datas/SAR-TZ-ship'
    #data_path ='/media/liu/088d8f6e-fca3-4aed-871f-243ad962413b/datas/car_det_train/val'
    #data_path = '/media/liu/088d8f6e-fca3-4aed-871f-243ad962413b/datas/dota1.5/generated_car'
    # data_path = '/media/data4T/datas/SAR-AIR-SARShip'
    data_path = '/data4T/datas/dota1.5/isAID-hull3/train' #
    total_train = generate_patches(data_path, 36, 180, 1.1, MIN_DISTANCE_TO_EDGE,
                                   256, 3, 1,output_folder='patches5',max_generated=0,draw_polys_on_big=10)
    
    data_path = '/data4T/datas/dota1.5/isAID-hull3/val' #
    total_val = generate_patches(data_path, 36, 180, 1.1, MIN_DISTANCE_TO_EDGE,
                                 256, 1, 1,output_folder='patches5',max_generated=0,draw_polys_on_big=10)
    
    print(f'\033[32mtotal_train={total_train} total_val={total_val}\033[0m')
