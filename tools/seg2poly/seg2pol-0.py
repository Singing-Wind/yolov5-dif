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
    return (M @ pts_h.T).T  # [N,2]

def process_image(img_path, seg_path, output_dir, diz=30, diz_angle=180, CROP_SIZE = (640, 640), MAX_SELECTION_PER_OBJ=1, least_len=3):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    image = cv2.imread(img_path) #image[H,W,3]
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE) #seg[H,W]
    assert image.shape[:2]==seg.shape
    H, W = image.shape[:2]
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    centers = []
    # Tracking how many times a contour is selected
    selection_counts = {}
    for i, c in enumerate(contours):
        if len(c) >= least_len:
            pts = c[:, 0, :]
            center = np.mean(pts, axis=0)
            key = f"{img_name}_{i}"
            valid_contours.append((key, pts))
            centers.append(center)
    
    output_count = 0
    for i, (key, pts) in enumerate(valid_contours):
        # 计数限制
        if selection_counts.get(key_j, 0) < MAX_SELECTION_PER_OBJ:
            center = centers[i]
            dx = np.random.uniform(-diz, diz)
            dy = np.random.uniform(-diz, diz)
            dst_offset = (CROP_SIZE[0] // 2 + dx, CROP_SIZE[1] // 2 + dy)
            angle = np.random.uniform(-diz_angle, diz_angle)
            M = get_affine_transform(center, 1.0, angle, dst_offset) #M[2,3]
            # M_inv = cv2.invertAffineTransform(M) #M_inv[2,3]

            # Warp original image
            cropped = cv2.warpAffine(image, M, CROP_SIZE, flags=cv2.INTER_LINEAR, borderValue=(0,0,0)) #cropped[h,w,3]
            out_img_path = os.path.join(output_dir, "images", f"{img_name}_{output_count}.jpg")
            cv2.imwrite(out_img_path, cropped)

            # Transform and filter contours
            txt_lines = []
            pol_lines = []
            # 3. 筛选目标
            # 1. 所有中心点先仿射变换 [N,2]
            centers_j = np.array([np.mean(pts_j, axis=0) for _, pts_j in valid_contours])  # [N,2]
            centers_affine = apply_affine(centers_j, M)  # [N,2]

            # 2. 过滤在CROP_SIZE范围内的目标
            in_bounds = (centers_affine >= 0) & (centers_affine < CROP_SIZE)  # [N,2] bool
            valid_mask = np.all(in_bounds, axis=1)  # [N] bool
            for j in np.where(valid_mask)[0]:  # 只处理在范围内的目标
                key_j, pts_j = valid_contours[j]                
            
                selection_counts[key_j] = selection_counts.get(key_j, 0) + 1

                # 点变换 + AABB
                pts_affine = apply_affine(pts_j, M)
                x_min, y_min = pts_affine.min(axis=0)
                x_max, y_max = pts_affine.max(axis=0)

                # 写 .txt（归一化框）
                xc = (x_max + x_min) / 2 / CROP_SIZE[0]
                yc = (y_max + y_min) / 2 / CROP_SIZE[1]
                w = (x_max - x_min) / CROP_SIZE[0]
                h = (y_max - y_min) / CROP_SIZE[1]
                txt_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

                # 写 .pol（归一化轮廓）
                pts_norm = pts_affine.copy()
                pts_norm[:, 0] /= CROP_SIZE[0]
                pts_norm[:, 1] /= CROP_SIZE[1]
                flat_coords = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in pts_norm])
                pol_lines.append(f"0 {flat_coords}")

            # Save .txt and .pol
            if txt_lines:
                with open(os.path.join(output_dir, "labels", f"{img_name}_{output_count}.txt"), "w") as f:
                    f.write("\n".join(txt_lines))
                with open(os.path.join(output_dir, "labels", f"{img_name}_{output_count}.pol"), "w") as f:
                    f.write("\n".join(pol_lines))
                output_count += 1

if __name__ == "__main__":
    # Constants
    CROP_SIZE = (640, 640)
    MAX_SELECTION_PER_OBJ = 5
    diz = 30  # Max random offset in pixels
    diz_angle = 180

    # Paths
    data_path = '/media/liu/088d8f6e-fca3-4aed-871f-243ad962413b/datas/remote_change'
    image_dir = os.path.join(data_path,"images")
    seg_dir =  os.path.join(data_path,"seg")
    #
    output_dir = None #"output3"
    if output_dir is None or not os.path.exists(output_dir):
        output_dir = os.path.join(data_path,"output")
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    # Run all
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    for img_file in tqdm(image_files):
        img_path = os.path.join(image_dir, img_file)
        seg_path = os.path.join(seg_dir, img_file)
        if os.path.exists(seg_path):
            process_image(img_path, seg_path, output_dir,
                          diz = diz, diz_angle=diz_angle, CROP_SIZE = CROP_SIZE, MAX_SELECTION_PER_OBJ=MAX_SELECTION_PER_OBJ,
                          least_len=3,
                          numpy_filter=1)
    # 写入类别名到 names.txt
    with open(os.path.join(output_dir, "names.txt"), "w") as f:
        f.write("change\n")

