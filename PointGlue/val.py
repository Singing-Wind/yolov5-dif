import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   # 设置中文字体（黑体）
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号
from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PointGlue.matchviz import plot_images_and_matches

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
from estimate import estimate_affine,estimate_affine_with_rotations
from metrics import affine_cosine_similarity
from np2image import np2image

from glob import glob
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint', filter_threshold=0.1).eval().to(device)

    data_path = '/media/liu/088d8f6e-fca3-4aed-871f-243ad962413b/datas/dota1.5/changes-patches'
    image_dir = os.path.join(data_path, 'images')
    image0_dir = os.path.join(data_path, 'images0')
    label_dir = os.path.join(data_path, 'labels')

    # 确保目录存在
    match_wrong_path = 'PointGlue/match_wrong'
    os.makedirs(match_wrong_path, exist_ok=True)

    image_files = sorted(glob(os.path.join(image_dir, '*.jpg')))
    results = []

    least_match,least_sim = 10, 0.99
    n_total,n_success = 0,0
    inlier_sum,simcos_sum = 0,0
    ransac_thresh = 3.0
    method = 1
    with tqdm(image_files, desc="Processing images", file=sys.stdout) as pbar:
        for i,fimage1 in enumerate(pbar):
            # if i>60:
            #     break
            name = os.path.splitext(os.path.basename(fimage1))[0]
            fimage0 = os.path.join(image0_dir, f'{name}.jpg')
            flabel = os.path.join(label_dir, f'{name}.npy')

            if (os.path.exists(fimage0) and os.path.exists(flabel)):
                try:
                    # 加载图像和标签
                    image0 = load_image(fimage0).to(device) #image0[C,H,W]
                    image1 = load_image(fimage1).to(device) #image1[C,H,W]
                    A23_gt = np.load(flabel) #A23_gt[2,3]

                    # 匹配估计
                    A23, n_matches, inliers, _, _ = estimate_affine_with_rotations(image0, image1, extractor, matcher,ransac_thresh) if method else estimate_affine(image0, image1, extractor, matcher,ransac_thresh)

                    if A23 is not None and inliers is not None:
                        n_inliers = int(np.sum(inliers))
                        sim_cos = affine_cosine_similarity(A23, A23_gt)
                    else:
                        n_inliers = 0
                        sim_cos = 0
                    results.append((n_matches, n_inliers, sim_cos))
                    #
                    n_total += 1
                    if n_inliers > least_match and sim_cos > least_sim:
                        n_success += 1
                    else:
                        if method:
                            img0 = np2image(image0.cpu().numpy())
                            img1 = np2image(image1.cpu().numpy())
                            vis = plot_images_and_matches(img0, img1, None, None)
                            # 构造保存路径
                            save_path = os.path.join(match_wrong_path, f'{name}_vis.jpg')
                            # 转换 RGB → BGR 再保存
                            cv2.imwrite(save_path, vis[:, :, ::-1])  # RGB → BGR
                    inlier_sum += n_inliers
                    simcos_sum += sim_cos
                    # 进度条右侧显示当前统计
                    pbar.set_postfix({
                        "n_inliers": n_inliers,
                        "sim_cos": f"{sim_cos:.4f}",
                        "success_rate": f"{100.0 * n_success / n_total:.1f}%" if n_total > 0 else "0%",
                        "avg_sim": f"{simcos_sum / n_total:.4f}" if n_total > 0 else "0.0000"
                    })
                except Exception as e:
                    print(f"Error processing {name}: {e}")
                    continue

    # 统计指标
    n_matches_ary = np.array([r[0] for r in results])
    n_inliers_ary = np.array([r[1] for r in results])
    sim_cos_ary = np.array([r[2] for r in results]).clip(-1.00,1.00)
    valid_mask = (n_inliers_ary > least_match) & (sim_cos_ary > least_sim)
    success_count = np.sum(valid_mask)
    success_ratio = success_count / len(results)

    # 展示统计信息
    print(f'\033[32msuccess_ratio={100*success_ratio:.5f}%={success_count}/{len(results)}\033[0m')

    # 创建保存目录
    os.makedirs('PointGlue/stats', exist_ok=True)

    # 1. n_matches 分布
    plt.figure(figsize=(6, 4))
    plt.hist(n_matches_ary, bins=30, color='steelblue', edgecolor='black')
    plt.title(f'Distribution of n_matches. avg_n_matches = {n_matches_ary.mean()}')
    plt.xlabel('n_matches')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('PointGlue/stats/hist_n_matches.png')
    plt.close()

    # 2. n_inliers 分布
    plt.figure(figsize=(6, 4))
    plt.hist(n_inliers_ary, bins=30, color='orange', edgecolor='black')
    plt.title(f'Distribution of n_inliers. avg_n_inliers = {n_inliers_ary.mean()}')
    plt.xlabel('n_inliers')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('PointGlue/stats/hist_n_inliers.png')
    plt.close()

    # 3. sim_cos 分布
    plt.figure(figsize=(6, 4))
    plt.hist(sim_cos_ary, bins=30, range=(-1.00, 1.00), color='green', edgecolor='black')
    plt.title(f'Distribution of sim_cos. avg_sim_cos = {sim_cos_ary.mean()}')
    plt.xlabel('sim_cos')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('PointGlue/stats/hist_sim_cos.png')
    plt.close()

    print("Histograms saved in stats/ folder.")
