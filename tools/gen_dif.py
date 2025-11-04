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

def mulA23(dA, A0):
    # dA,A0: 2×3 仿射矩阵
    R0, t0 = A0[:, :2], A0[:, 2]
    Rd, td = dA[:, :2], dA[:, 2]
    
    R = Rd @ R0                # 新的旋转缩放部分
    t = Rd @ t0 + td           # 新的平移部分
    
    return np.hstack([R, t[:, None]]).astype(np.float32)  # 合成2×3矩阵

def invA23(A):
    # 求 2×3 仿射矩阵的逆。
    # 输入:
    #     A: np.ndarray of shape (2, 3)
    # 返回:
    #     A_inv: np.ndarray of shape (2, 3)
    R, t = A[:, :2], A[:, 2]
    R_inv = np.linalg.inv(R)
    t_inv = -R_inv @ t
    return np.hstack([R_inv, t_inv[:, None]]).astype(np.float32)

def is_center_inside(p0_rev, img_size):
    (W, H) = img_size
    # 1. 计算中心点坐标
    center = np.mean(p0_rev, axis=0)  # shape: (2,), [x, y]

    # 2. 拆出 x, y
    cx, cy = center

    # 3. 判断是否在图像边界内（严格小于 W,H 且 ≥ 0）
    inside = 0 <= cx < W and 0 <= cy < H

    return inside, center  # 可以返回 bool 和中心点坐标

def add_sparse_noise(image: np.ndarray, ratio: float = 0.1, noise_range: int = 5) -> np.ndarray:
    # 给图像添加稀疏噪声：仅改变一部分像素的值，噪声值在 [-noise_range, noise_range] 内。
    # 参数:
    #     image (np.ndarray): 原始图像，支持灰度或彩色图像，类型应为 np.uint8。
    #     ratio (float): 被添加噪声的像素比例（0~1）。
    #     noise_range (int): 噪声的最大幅度，范围为 [-noise_range, +noise_range]。
    # 返回:
    #     np.ndarray: 添加了稀疏噪声的新图像。
    assert image.dtype == np.uint8, "图像类型必须是 uint8"
    h, w = image.shape[:2]
    c = 1 if image.ndim == 2 else image.shape[2]

    # 初始化噪声数组
    noise = np.zeros((h, w, c), dtype=np.int8)

    # 随机选择需要添加噪声的像素位置
    num_pixels = int(ratio * h * w)
    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)

    # 给选中的位置添加随机噪声
    for y, x in zip(ys, xs):
        rand_noise = np.random.randint(-noise_range, noise_range + 1, size=(c,), dtype=np.int8)
        noise[y, x] = rand_noise

    # 如果是灰度图像，去掉多余的通道维度
    if image.ndim == 2:
        noise = noise[:, :, 0]

    # 添加噪声，防止溢出
    noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_image


# xc = random.randint(a, w - a)
# yc = random.randint(b, h - b)
# a = random.randint(min_ab, max_ab)
# b = random.randint(min_ab, max_ab)
# 提取 ROI 区域并处理（加噪声或模糊）
# op = random.choice(['blur', 'noise'])
def process_multiple_ellipse_patches(image, n_ellipse=5, a_range=(20, 50), b_range=(10, 30), op_choices=('blur', 'noise'), edge = 2, noise_rate=0.3, noise_range=30):
    # 在图像中随机生成多个旋转椭圆区域，随机执行模糊或噪声处理，返回处理后的图像和所有区域的旋转矩形坐标点。 
    # 参数：
    #     image: 输入图像 (H, W, 3)
    #     n_ellipse: 椭圆数量
    #     a_range, b_range: 主轴与副轴长度范围
    #     op_choices: 可选操作，'blur' 或 'noise' 
    # 返回：
    #     image_copy: 处理后的图像
    #     pts_list: 每个椭圆的旋转矩形 4 点列表，List[np.ndarray shape=(4, 2)]

    h, w = image.shape[:2]
    image_copy = image.copy()
    pts_list,txt_lst = [],[]

    mask_blur = np.zeros((h, w), dtype=np.uint8)
    mask_noise = np.zeros((h, w), dtype=np.uint8)

    for _ in range(n_ellipse):
        # 随机中心点、轴长和角度
        a = random.randint(*a_range)
        b = random.randint(*b_range)
        xc = random.randint(edge, w - edge)
        yc = random.randint(edge, h - edge)
        theta_deg = random.uniform(0, 360)
        op = random.choice(op_choices)

        # 创建椭圆mask
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (int(xc), int(yc))
        axes = (int(a), int(b))
        if op == 'blur':
            cv2.ellipse(mask_blur, center, axes, theta_deg, 0, 360, 255, -1)
            obj_id = 0
        elif op == 'noise':
            cv2.ellipse(mask_noise, center, axes, theta_deg, 0, 360, 255, -1)
            obj_id = 1
        # if op == 'blur':
        #     blurred = cv2.GaussianBlur(image, (11, 11), sigmaX=5)
        #     image_copy[mask == 255] = blurred[mask == 255]
        # elif op == 'noise':
        #     noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        #     noisy = cv2.add(image, noise)
        #     image_copy[mask == 255] = noisy[mask == 255]

        # 计算椭圆外接旋转矩形
        ellipse = ((xc, yc), (2*a, 2*b), theta_deg)
        pts = cv2.boxPoints(ellipse)  # shape: (4, 2)
        pts_list.append(pts.astype(np.float32))

        # 计算最小外接矩形的中心、宽高
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        xmin, xmax = x_coords.min(), x_coords.max()
        ymin, ymax = y_coords.min(), y_coords.max()
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin

        # 保存为 [id, x, y, w, h]
        txt_lst.append([obj_id, x_center/w, y_center/h, width/w, height/h])
    
    # 最后统一生成两个处理图像
    blurred = cv2.GaussianBlur(image, (11, 11), sigmaX=5)
    # noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    # noisy = cv2.add(image, noise)
    noisy = add_sparse_noise(image, noise_rate, noise_range)

    # 再分别合成
    image_copy = image.copy()
    image_copy[mask_blur == 255] = blurred[mask_blur == 255]
    image_copy[mask_noise == 255] = noisy[mask_noise == 255]

    return image_copy, txt_lst, pts_list



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

colors = [
        (54, 67, 244),
        (233, 30, 99),
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

def save_patch(image0, image1, label_txt, label_pts, patch_name, output_path, txt0_lst, label_pts0, img_size, draw_pts=[]):
    assert len(label_txt) ==0 or len(label_txt) == len(label_pts)

    txt_name = f"{output_path}/labels/{patch_name}.txt"
    pts_name = f"{output_path}/labels/{patch_name}.pts"

    if len(draw_pts) > 0:
        assert len(draw_pts)==2
        assert len(label_pts0)==len(label_pts)
        for i,p in enumerate(label_pts0):
            id = int(txt0_lst[i][0])
            # p_scaled = p.copy()
            # p_scaled[:, 0] *= img_size[0]  # 缩放 x 分量
            # p_scaled[:, 1] *= img_size[1]  # 缩放 y 分量
            # 转为 int32，并 reshape 成 OpenCV 需要的格式
            if draw_pts[0]:
                pts = p.reshape((-1, 1, 2)).astype(np.int32)
                # pts = p.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(image0,[pts],True,colors[id%len(colors)])
            if draw_pts[1]:
                pts = label_pts[i].reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(image1,[pts],True,colors[id%len(colors)])

    image_name = f"{output_path}/images0/{patch_name}.jpg"
    cv2.imwrite(image_name, image0)
    image_name1 = f"{output_path}/images/{patch_name}.jpg"
    cv2.imwrite(image_name1, image1)

    if len(label_pts) > 0:
        with open(txt_name, "w") as f:
            for line in label_txt:
                obj_id = int(line[0])
                x = line[1] / img_size[0]
                y = line[2] / img_size[1]
                w = line[3] / img_size[0]
                h = line[4] / img_size[1]
                f.write(f"{obj_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        with open(pts_name, "w") as f:
            for line in label_pts:
                flat_line = line.flatten()
                norm_line = [
                    flat_line[i] / img_size[0] if i % 2 == 0 else flat_line[i] / img_size[1]
                    for i in range(8)
                ]
                f.write(" ".join(f"{v:.6f}" for v in norm_line) + "\n")

def generate_changes(data_path, diz, diz_angle, diz_scale, min_dist_to_edge, select_rate=1.0, output_folder='changes-patches',max_gen_count=0,draw_pts=[0,0]):
    image_dir = os.path.join(data_path, 'images')
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'))]

    output_path = os.path.join(data_path, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    change_names = ['blur', 'noise'] #
    names_file = os.path.join(data_path, output_folder, 'names.txt')
    with open(names_file, 'w') as f:
        for name in change_names:
            f.write(name + '\n')

    if not os.path.exists(os.path.join(output_path, 'images')):
        os.makedirs(os.path.join(output_path, 'images'))
    if not os.path.exists(os.path.join(output_path, 'images0')):
        os.makedirs(os.path.join(output_path, 'images0'))
    if not os.path.exists(os.path.join(output_path, 'labels')):
        os.makedirs(os.path.join(output_path, 'labels'))

    total_img = 0
    for image_path in tqdm(image_files, total=len(image_files)):
        image, txt_data, pts_data = read_image_label(image_path)
        if image is None or txt_data is None or pts_data is None:
            print(f"Failed to read {image_path}")
            continue

        height, width = image.shape[:2]

        filename_with_extension = os.path.basename(image_path)
        filename, _ = os.path.splitext(filename_with_extension)
            
        #gen negitive samples
        max_neg_select = max(round(select_rate*(height*width/(PATCH_SIZE[0]*PATCH_SIZE[1]))),1)
        for i in range(max_neg_select):
            x_center, y_center = random.randint(min_dist_to_edge, width - min_dist_to_edge), random.randint(min_dist_to_edge, height - min_dist_to_edge)
            angle = random.uniform(-diz_angle,diz_angle)
            A0 = get_affine_transform((x_center, y_center), scale * random.uniform(1.0/diz_scale, diz_scale),angle,(PATCH_SIZE[0]/2,PATCH_SIZE[1]/2))
            dA = get_affine_transform((PATCH_SIZE[0]/2 + random.uniform(-diz,diz), PATCH_SIZE[1]/2 + random.uniform(-diz,diz)),
                                      scale * random.uniform(1.0/diz_scale, diz_scale),
                                      angle,
                                      (PATCH_SIZE[0]/2,PATCH_SIZE[1]/2))
            A1 = mulA23(dA,A0)
            patch0 = cv2.warpAffine(image, A0, PATCH_SIZE)
            patch1 = cv2.warpAffine(image, A1, PATCH_SIZE)
            dAT = invA23(dA)

            #---------------------------
            edge = 2
            xc = random.randint(edge, PATCH_SIZE[0] - edge)
            yc = random.randint(edge, PATCH_SIZE[1] - edge)
            ab_range=(15,120)
            a = random.randint(ab_range[0], ab_range[1])
            b = random.randint(ab_range[0], ab_range[1])
            theta_deg = random.uniform(-180, 180)
            # 提取 ROI 区域并处理（加噪声或模糊）
            # _, pts = process_multiple_ellipse_patches(patch0, n_ellipse=random.randint(1,4), a_range=(20, 120), b_range=(20, 120), op_choices=None, edge = 2)
            patch1,txt1_lst,pts1_lst = process_multiple_ellipse_patches(patch1, n_ellipse=random.randint(1,4), a_range=(20, 120), b_range=(20, 120), op_choices=change_names, edge = 2)
            # cv2.fillPoly(patch0, [np.int32(new_pts * PATCH_SIZE)], (128, 128, 128))
            pts0_lst,txt0_lst, insides = [],[],[]
            for j,p in enumerate(pts1_lst):
                # Step 1: 转置
                p_T = p.T  # shape: (2, 4)
                # Step 2: 添加齐次维度
                ones = np.ones((1, p_T.shape[1]), dtype=np.float32)  # shape: (1, 4)
                p_hom = np.vstack([p_T, ones])  # shape: (3, 4)
                # Step 3: 左乘仿射变换
                p0_rev = (dAT @ p_hom).T  # shape: (2, 4)->p0_rev[4,2]
                pts0_lst.append(p0_rev)

                # 计算最小外接矩形的中心、宽高
                x_coords = p0_rev[:, 0]
                y_coords = p0_rev[:, 1]
                xmin, xmax = x_coords.min(), x_coords.max()
                ymin, ymax = y_coords.min(), y_coords.max()
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin

                # 保存为 [id, x, y, w, h]
                obj_id = txt1_lst[j][0]
                txt0_lst.append([obj_id, x_center, y_center, w, h])

                inside, center = is_center_inside(p0_rev,PATCH_SIZE)
                insides.append(inside)
            
            assert len(pts1_lst)==len(txt1_lst) and len(pts1_lst)==len(pts0_lst) and len(pts1_lst)==len(txt0_lst)
            
            pts1_lst = [p for p, keep in zip(pts1_lst, insides) if keep]
            txt1_lst = [p for p, keep in zip(txt1_lst, insides) if keep]
            pts0_lst = [p for p, keep in zip(pts0_lst, insides) if keep]
            txt0_lst = [t for t, keep in zip(txt0_lst, insides) if keep]

            assert len(pts1_lst)==len(txt1_lst) and len(pts1_lst)==len(pts0_lst) and len(pts1_lst)==len(txt0_lst)

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            patch_name = f'{base_name}_{x_center}_{y_center}-{i}'
            save_patch(patch0, patch1, txt1_lst, pts1_lst, patch_name, output_path, txt0_lst, pts0_lst,PATCH_SIZE,draw_pts=draw_pts)
            #save dA
            a23_name = f"{output_path}/labels/{patch_name}.npy"
            np.save(a23_name, dA)
            
        total_img+=1
        if max_gen_count>0 and total_img>max_gen_count:
            break

if __name__ == '__main__':
    # 示例用法
    # data_path = '/media/data4T/datas/SAR-TZ-ship'
    #data_path ='/media/liu/088d8f6e-fca3-4aed-871f-243ad962413b/datas/car_det_train/val'
    #data_path = '/media/liu/088d8f6e-fca3-4aed-871f-243ad962413b/datas/dota1.5/generated_car'
    # data_path = '/media/data4T/datas/SAR-AIR-SARShip'
    data_path = '/media/liu/088d8f6e-fca3-4aed-871f-243ad962413b/datas/dota1.5'
    generate_changes(data_path, 36, 180, 1.2, MIN_DISTANCE_TO_EDGE, 0.002, output_folder='changes-val',max_gen_count=600,draw_pts=[0,0])
