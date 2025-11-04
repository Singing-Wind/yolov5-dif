import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   # 设置中文字体（黑体）
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号
from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image
#download LightGlue
#pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
#/home/liu/.cache/torch/hub/checkpoints
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PointGlue.matchviz import plot_images_and_matches

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
from estimate import estimate_affine,estimate_affine_with_rotations
from metrics import affine_cosine_similarity
from np2image import np2image

if __name__ == "__main__":
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint', filter_threshold=0.1).eval().to(device)

    data_path = '/sgg/liujin/workspace/datas/LEVIR-CD4/val' #'/media/liu/088d8f6e-fca3-4aed-871f-243ad962413b/datas/dota1.5/changes-patches'
    image_dir = os.path.join(data_path, 'images')
    image0_dir = os.path.join(data_path, 'images0')
    label_dir = os.path.join(data_path, 'labels')

    # name = 'P2770_571.1870727539062_244.161865234375-1' #fail
    name = 'val_17_2' #'P2756_232.01312255859375_213.1663818359375-0' #fail ++++++
    # name = 'P2755_234.213134765625_162.68568420410156-1' #fail
    # name = 'P2805_604.8345947265625_562.8802490234375-0'
    fimage0 = os.path.join(image0_dir,f'{name}.jpg')
    assert os.path.exists(fimage0)
    fimage1 = os.path.join(image_dir,f'{name}.jpg')
    assert os.path.exists(fimage1)
    flabel = os.path.join(label_dir,f'{name}.npy')
    assert os.path.exists(flabel)
    # 2. 加载图像（640x640，灰度）
    image0 = load_image(fimage0).to(device) #image0[C,H,W]
    image1 = load_image(fimage1).to(device) #image1[C,H,W]

    A23_gt = np.load(flabel)

    ransac_thresh = 4.0
    if 1:
        A23, n_matches, inliers, mkpts0, mkpts1 = estimate_affine_with_rotations(
            image0, image1, extractor, matcher, ransac_thresh=ransac_thresh
        )
    else:
        A23, n_matches, inliers, mkpts0, mkpts1 = estimate_affine(image0, image1, extractor, matcher, ransac_thresh=ransac_thresh)

    # 7. 输出变换矩阵
    n_inliers = int(np.sum(inliers))
    print(f"匹配数量: {n_matches}, 内点数: {n_inliers}")
    print("pred仿射矩阵 A23:")
    print(A23)
    print("gt仿射矩阵 A23:")
    print(A23_gt)
    sim_cos = affine_cosine_similarity(A23,A23_gt)
    print(sim_cos)

    # 8. 提取旋转、缩放、平移参数
    a11, a12 = A23[0, 0], A23[0, 1]
    a21, a22 = A23[1, 0], A23[1, 1]
    tx, ty = A23[0, 2], A23[1, 2]

    rotation_deg = np.arctan2(a21, a11) * 180 / np.pi
    scale = (np.hypot(a11, a21) + np.hypot(a12, a22)) / 2

    print(f"\n参数解析:")
    print(f"→ 旋转角度: {rotation_deg:.2f}°")
    print(f"→ 缩放因子: {scale:.3f}")
    print(f"→ 平移向量: tx = {tx:.1f}, ty = {ty:.1f}")

    # 9. 可视化匹配结果
    # 转为 [H, W, C]
    img0 = np2image(image0.cpu().numpy())
    img1 = np2image(image1.cpu().numpy())
    # img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    inlier_mask = inliers.ravel() > 0
    mkpts0_in = mkpts0[inlier_mask]
    mkpts1_in = mkpts1[inlier_mask]

    vis = plot_images_and_matches(img0, img1, mkpts0_in, mkpts1_in)

    if 1:
        plt.figure(figsize=(12, 6))
        plt.imshow(vis)
        plt.axis('off')
        plt.title('SuperPoint + LightGlue 匹配（内点）')
        plt.tight_layout()
        # 保存图片
        plt.savefig("match_result.png", dpi=500, bbox_inches='tight')  # 保存为 PNG，高分辨率
        plt.close()
        # plt.show() #block=False
        # plt.close()
    else:
        cv2.imshow("匹配结果", vis[:, :, ::-1])  # RGB → BGR
        cv2.waitKey(0)  # 等待按键，0 表示无限等待
        cv2.destroyAllWindows()
