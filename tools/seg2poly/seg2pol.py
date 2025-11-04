import os
import cv2
import numpy as np
from tqdm import tqdm

# Set random seed
np.random.seed(88)

def get_affine_transform(center, s, angle, dst_offset):
    angle_rad = np.deg2rad(angle)
    cos_rad = np.cos(angle_rad)
    sin_rad = np.sin(angle_rad)
    cx, cy = center
    dx, dy = dst_offset
    transform_matrix = np.array([
        [ s * cos_rad, s * sin_rad, dx - s * ( cos_rad * cx + sin_rad * cy)],
        [-s * sin_rad, s * cos_rad, dy - s * (-sin_rad * cx + cos_rad * cy)]
    ])
    return transform_matrix

def apply_affine(pts, M):
    """Apply affine transform to a set of points [N,2]"""
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])  # to homogeneous
    return (M @ pts_h.T).T  # [n,2]


def compute_dA23(A23_1, A23_0):
    # 扩展为 3x3 齐次矩阵
    A1 = np.vstack([A23_1, [0, 0, 1]])
    A0 = np.vstack([A23_0, [0, 0, 1]])

    # 计算复合变换
    dA = A1 @ np.linalg.inv(A0)

    # 截取前两行，得到 2x3 仿射矩阵
    return dA[:2, :]

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed

def process_image(img_path, seg_path, diz=30, diz_angle=180, CROP_SIZE = (640, 640), MAX_SELECTION_PER_OBJ=1, least_len=3, sim=1, hull=1, hyp={}):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    image = cv2.imread(img_path) #image[H,W,3] 不能有中文
    if sim==0:
        image0 = cv2.imread(os.path.join(os.path.dirname(os.path.dirname(img_path)),'images0',img_name+'.tif'))
    else:
        image0 = image.copy()
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE) #seg[H,W]
    assert image.shape[:2]==seg.shape
    H, W = image.shape[:2]
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tracking how many times a contour is selected
    select_pols = [] #select_pols[[npts,2],..]
    select_keys = []
    centers = []
    for i, poly in enumerate(contours):
        if len(poly) >= least_len:
            pts = poly[:, 0, :]
            if hull:
                # pts = np.array(pts, dtype=np.float32).reshape(-1, 2)        # [n, 2]               
                pts = cv2.convexHull(pts).reshape(-1, 2)
            select_pols.append(pts)
            center = np.mean(pts, axis=0)
            key = f"{img_name}_{i}"
            select_keys.append(key)
            centers.append(center)
    cv2.fillPoly(image0, select_pols, color=(128, 128, 128)) #select_pols[[npts,2],..]
    
    output_count = 0
    selection_counts = {}
    assert len(select_keys)==len(select_pols)
    for i, (key, pts) in enumerate(zip(select_keys,select_pols)):
        # 计数限制
        if selection_counts.get(key, 0) < MAX_SELECTION_PER_OBJ:
            center = centers[i]
            dx = np.random.uniform(-diz, diz)
            dy = np.random.uniform(-diz, diz)
            angle = np.random.uniform(-diz_angle, diz_angle)
            dst_offset = (CROP_SIZE[0] // 2 + dx, CROP_SIZE[1] // 2 + dy)
            M = get_affine_transform(center, 1.0, angle, dst_offset) #M[2,3]
            # M_inv = cv2.invertAffineTransform(M) #M_inv[2,3]

            # Warp original image
            cropped = cv2.warpAffine(image, M, CROP_SIZE, flags=cv2.INTER_LINEAR, borderValue=(0,0,0)) #cropped[h,w,3]
            augment_hsv(cropped, hgain=hyp.get('hsv_h',0.015), sgain=hyp.get('hsv_s',0.7), vgain=hyp.get('hsv_v',0.4))
            out_img_path = os.path.join(output_dir, "images", f"{img_name}_{output_count}.jpg")
            cv2.imwrite(out_img_path, cropped)

            # Transform and filter contours
            pol_lines = []
            # 3. 筛选目标
            # 1. 所有中心点先仿射变换 [n,2]
            centers_j = np.array([np.mean(pol_j, axis=0) for pol_j in select_pols])  # [n,2]
            centers_affine = apply_affine(centers_j, M)  # [n,2]

            # 2. 过滤在CROP_SIZE范围内的目标
            in_bounds = (centers_affine >= 0) & (centers_affine < CROP_SIZE)  # [n,2] bool
            valid_mask = np.all(in_bounds, axis=1)  # [n] bool
            for j in np.where(valid_mask)[0]:  # 只处理在范围内的目标
                key_j = select_keys[j]
                pol_j = select_pols[j] #pol_j[npts,2]
            
                selection_counts[key_j] = selection_counts.get(key_j, 0) + 1

                pts_affine = apply_affine(pol_j, M).astype(np.float32) # pts_affine[n,2], dtype是

                # 写 .pol（归一化轮廓）
                flat_coords = " ".join([f"{p[0]/CROP_SIZE[0]:.6f} {p[1]/CROP_SIZE[1]:.6f}" for p in pts_affine])
                pol_lines.append(f"0 {flat_coords}")

            # Save .pol
            if len(pol_lines) > 0:
                with open(os.path.join(output_dir, "labels", f"{img_name}_{output_count}.pol"), "w") as f:
                    f.write("\n".join(pol_lines))

            output_count += 1

if __name__ == "__main__":
    # Constants
    CROP_SIZE = (640, 640)
    MAX_SELECTION_PER_OBJ = 5
    diz = 30  # Max random offset in pixels

    # Paths
    image_dir = '/home/liu/workspace/datas/WHU/images' #"images"
    data_path = os.path.dirname(image_dir)
    seg_dir = os.path.join(data_path,'seg')# "seg"
    output_dir = os.path.join(data_path,'patches_pol') #"output3"
    if os.path.exists(output_dir):
        print(f'\033[31m{output_dir} already exists!\033[0m')
        response = input("⚠️ 警告：是否继续？(y/n): ").strip().lower()
        if response == 'y':
            print("继续运行...")
            # 继续你的代码逻辑
        else:
            print("已终止程序。")
            exit(0)  # 或者 sys.exit(0)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    # Run all
    hyp={'hsv_h':0.015,'hsv_s':0.7,'hsv_v':0.4}
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    for img_file in tqdm(image_files):
        img_path = os.path.join(image_dir, img_file)
        seg_path = os.path.join(seg_dir, img_file)
        if os.path.exists(seg_path):
            process_image(img_path, seg_path,diz = diz, CROP_SIZE = CROP_SIZE, MAX_SELECTION_PER_OBJ=MAX_SELECTION_PER_OBJ, least_len=3, hull=1, hyp=hyp)
    # 写入类别名到 names.txt
    with open(os.path.join(output_dir, "names.txt"), "w") as f:
        f.write("change\n")

