import os
import json
import cv2
import shutil
import numpy as np
from tqdm import tqdm
# from pycocotools import mask as maskUtils

# =============================
# 傅里叶系数计算函数
# =============================
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
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    a0 = 1. / N * np.sum(x)
    c0 = 1. / N * np.sum(y)
    an, bn, cn, dn = [np.zeros(terms + 1) for _ in range(4)]
    for k in range(1, (N // 2) + 1):
        if k > terms:
            break
        an[k] = 2. / N * np.sum(x * np.cos(k * t))
        bn[k] = 2. / N * np.sum(x * np.sin(k * t))
        cn[k] = 2. / N * np.sum(y * np.cos(k * t))
        dn[k] = 2. / N * np.sum(y * np.sin(k * t))
    if return_list:
        list_coef = [a0, c0]
        for k in range(1, terms + 1):
            list_coef.extend([an[k], bn[k], cn[k], dn[k]])
        return list_coef
    an[0] = a0
    cn[0] = c0
    return (an, bn, cn, dn), (x, y)
def fft_1(coeffs,npts):
    n = (len(coeffs) - 2) // 4
    t = np.linspace(0, 1, npts, endpoint=False)
    x = np.ones_like(t) * coeffs[0]
    y = np.ones_like(t) * coeffs[1]
    offset = 2
    for i in range(n):#i==0其实就是1阶了，所以i到n-1其实就是n阶
        assert offset + 4 <= len(coeffs)
        a, b, c, d = coeffs[offset : offset + 4]
        x += a * np.cos(2*np.pi*(i+1)*t) + b * np.sin(2*np.pi*(i+1)*t) #i==0其实就是1阶了，所以要i+1
        y += c * np.cos(2*np.pi*(i+1)*t) + d * np.sin(2*np.pi*(i+1)*t)
        offset+=4
    return x,y

def ft_scale(ft,scale):
    assert ((len(ft)-2)%4) == 0
    ft_terms = (len(ft)-2) / 4
    scalex,scaley = scale
    # 处理 0阶项
    ft[0] *= scalex   # a0 (x方向)
    ft[1] *= scaley  # c0 (y方向)

    if 1:
        ft[2::4] *= scalex   # a_i
        ft[3::4] *= scalex   # b_i
        ft[4::4] *= scaley  # c_i
        ft[5::4] *= scaley  # d_i
    else:
        # 处理 1阶及以上
        for i in range(ft_terms):
            base = 2 + i * 4  # 每阶开始的位置
            ft[base + 0] *= scalex  # a_i
            ft[base + 1] *= scalex  # b_i
            ft[base + 2] *= scaley  # c_i
            ft[base + 3] *= scaley  # d_i

# =============================
# polygon to YOLO bbox
# =============================
def polygon_to_yolo(x, y, width, height):
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    x_center = (x_min + x_max) / 2.0 / width
    y_center = (y_min + y_max) / 2.0 / height
    w = (x_max - x_min) / width
    h = (y_max - y_min) / height
    return x_center, y_center, w, h

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
def add_suffix_to_filename(path, suffix, ext2=None):
    # 分离路径和文件名
    dir_name, base_name = os.path.split(path)
    # 分离文件名和扩展名
    file_name, ext = os.path.splitext(base_name)
    # 添加后缀到文件名
    new_file_name = f"{file_name}{suffix}{ext}" if ext2==None else f"{file_name}{suffix}{ext2}"
    # 重新组合成新的路径
    new_path = os.path.join(dir_name, new_file_name)
    return new_path

# =============================
# 主函数：处理 COCO 标注和 mask，生成 yolo + ft
# =============================
def convert_isaid_subset(data_path, image_dir, annotation_file, output_dir, ft_terms=12, cp_img=True,npts=256,max_draw_poly=0,hull=1,min_pol_length=6):
    if data_path!='':
        image_dir = os.path.join(data_path,image_dir)
        # mask_dir = os.path.join(data_path,mask_dir)
        annotation_file = os.path.join(data_path,annotation_file)
        output_dir = os.path.join(data_path,output_dir)
    #
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    with open(annotation_file, 'r') as f:
        coco = json.load(f)

    # 创建 category_id 映射到从0开始的 class_id
    id2name = {cat['id']: cat['name'] for cat in coco['categories']}
    sorted_names = sorted(set(id2name.values()))
    name2idx = {name: idx for idx, name in enumerate(sorted_names)}
    id2idx = {cat_id: name2idx[name] for cat_id, name in id2name.items()}

    fnames = os.path.join(output_dir, "names.txt")
    with open(fnames, 'w', encoding='utf-8') as f:
        for name in sorted_names:
            f.write(name + '\n')

    image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}

    annotations_per_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        annotations_per_image.setdefault(img_id, []).append(ann)

    for img_id, filename in tqdm(image_id_to_filename.items(), desc="处理图像"):
        img_path = os.path.join(image_dir, filename)
        # mask_path = os.path.join(mask_dir, filename.replace('.png', '_instance_id_RGB.png'))
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            yolo_lines, ft_lines, pol_lines = [], [], []

            for ann in annotations_per_image.get(img_id, []):
                coco_cat_id = ann['category_id']
                class_id = id2idx[coco_cat_id]  # ✅ 转为从0开始
                seg = ann['segmentation']
                if isinstance(seg, list):  # polygon
                    for poly in seg:
                        if len(poly) >= min_pol_length*2:
                            # norm_poly = [poly[i] / width if i % 2 == 0 else poly[i] / height for i in range(len(poly))]
                            bbox = polygon_to_yolo(poly[0::2], poly[1::2], width, height) #bbox[4(xywh)]
                            yolo_lines.append(f"{class_id} {' '.join(f'{v:.6f}' for v in bbox)}")
                            # ✅ 输出归一化 polygon 坐标
                            norm_poly = [poly[i] / width if i % 2 == 0 else poly[i] / height for i in range(len(poly))]
                            pol_lines.append(f"{class_id} {' '.join(f'{v:.6f}' for v in norm_poly)}")
                            if hull:
                                poly = np.array(poly, dtype=np.float32).reshape(-1, 2)        # [n, 2]               
                                poly = cv2.convexHull(poly).reshape(-1, 2).flatten().tolist()
                            ft_lst = compute_coefficients_interp(poly, terms=ft_terms, return_list=True) #ft[2+4n]
                            ft = np.array(ft_lst, dtype=np.float32)
                            ft_scale(ft,(1.0/width,1.0/height))
                            ft_lines.append(f"{class_id} {' '.join(f'{v:.6f}' for v in ft)}")
                            #
                            if npts>0:
                                coeffs = ft.copy() #np.array(ft_lst,dtype=np.float32) #coeffs[2+4n]
                                ft_scale(coeffs,(width,height))
                                xy = fft_1(coeffs,npts)
                                #xy(x,y)->p[2,npts]->p[npts,2]
                                pts = np.vstack(list(xy)).T #pts[2,npts]->pts[npts,2]
                                pts = pts.reshape((-1, 1, 2)).astype(np.int32) #pts[npts,2]->pts[npts,1,2]
                                cv2.polylines(img,[pts],True,colors[class_id%len(colors)],thickness=2)
            if npts>0:#draw polys
                if max_draw_poly>0:
                    dst_img = os.path.join(output_dir, add_suffix_to_filename(filename,'_poly'))
                    cv2.imwrite(dst_img,img)
                    max_draw_poly-=1
            # 复制图像
            if cp_img:
                dst_img = os.path.join(output_dir, "images", filename)
                if not os.path.exists(dst_img):
                    shutil.copy(img_path, dst_img)

            label_file = filename.replace('.png', '.txt')
            with open(os.path.join(output_dir, "labels", label_file), 'w') as f:
                f.write('\n'.join(yolo_lines))
            ft_file = filename.replace('.png', '.ft')
            with open(os.path.join(output_dir, "labels", ft_file), 'w') as f:
                f.write('\n'.join(ft_lines))
            pol_file = filename.replace('.png', '.pol')
            with open(os.path.join(output_dir, "labels", pol_file), 'w') as f:
                f.write('\n'.join(pol_lines))

# =============================
# 示例调用
# =============================
if __name__ == '__main__':
    data_path = '/media/liu/f4854541-32b0-4d00-84a6-13d3a5dd30f2/datas/dota1.5'
    convert_isaid_subset(
        data_path = data_path,
        image_dir="images",
        # mask_dir="train/Instance_masks/images",
        annotation_file="isAID/train/Annotations/iSAID_train_2019.json",
        output_dir="isAID-hull3/train",
        ft_terms=12,
        npts=256,
        cp_img=True,
        max_draw_poly=6,
        hull=1
    )
    convert_isaid_subset(
        data_path = data_path,
        image_dir="val/images",
        # mask_dir="val/Instance_masks/images",
        annotation_file="isAID/val/Annotations/iSAID_val_2019.json",
        output_dir="isAID-hull3/val",
        ft_terms=12,
        npts=256,
        cp_img=True,
        max_draw_poly=6,
        hull=1
    )
