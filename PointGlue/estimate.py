import numpy as np
import torch
import cv2

def estimate_affine_with_rotations_feat(feats0, img1, extractor, matcher, ransac_thresh=3.0):
    # 对 img1 多旋转进行匹配估计仿射变换
    # 参数：
    #     feats0: torch.Tensor [C,H,W]
    #     img1: torch.Tensor [C,H,W]
    #     extractor: SuperPoint 特征提取器
    #     matcher: LightGlue 匹配器
    # 返回：
    #     A23: 估计的仿射矩阵 (2×3)，若失败则为 None
    #     n_matches: 合并后的匹配点对数量
    #     n_inliers: RANSAC 内点数量，若失败则为 0
    ROTATIONS = [0, 90, 180, 270]
    H, W = img1.shape[-2:]

    mkpts0_all = []
    mkpts1_all = []

    with torch.no_grad():
        for angle in ROTATIONS:
            if angle == 0:
                img1_rot = img1
            else:
                k = angle // 90
                img1_rot = torch.rot90(img1[None], k=k, dims=[2, 3])[0]

            feats1_rot = extractor.extract(img1_rot)
            matches = matcher({'image0': feats0, 'image1': feats1_rot})

            kpts0 = feats0['keypoints'][0].cpu().numpy()
            kpts1_rot = feats1_rot['keypoints'][0].cpu().numpy()
            matches0 = matches['matches0'][0].cpu().numpy()

            valid = matches0 > -1
            if np.sum(valid) > 0:
                mk0 = kpts0[valid]
                mk1_rot = kpts1_rot[matches0[valid]]

                if angle == 0:
                    mk1 = mk1_rot
                elif angle == 90:
                    # mk1 = np.stack([mk1_rot[:, 1], H - 1 - mk1_rot[:, 0]], axis=-1)
                    # 逆时针90° → 顺时针270° → (x', y') → (y, H - 1 - x)
                    mk1 = np.stack([H - 1 - mk1_rot[:, 1], mk1_rot[:, 0]], axis=-1)
                elif angle == 180:
                    # 逆时针180° → 顺时针180° → (x', y') → (W - 1 - x, H - 1 - y)
                    mk1 = np.stack([W - 1 - mk1_rot[:, 0], H - 1 - mk1_rot[:, 1]], axis=-1)
                elif angle == 270:
                    # mk1 = np.stack([W - 1 - mk1_rot[:, 1], mk1_rot[:, 0]], axis=-1)
                    # 逆时针270° → 顺时针90° → (x', y') → (y, W - 1 - x)
                    mk1 = np.stack([mk1_rot[:, 1], W - 1 - mk1_rot[:, 0]], axis=-1)

                mkpts0_all.append(mk0)
                mkpts1_all.append(mk1)

    if len(mkpts0_all) > 0:
        mkpts0 = np.concatenate(mkpts0_all, axis=0)
        mkpts1 = np.concatenate(mkpts1_all, axis=0)
        n_matches = len(mkpts0)

        A23, inliers = cv2.estimateAffinePartial2D(
            mkpts0, mkpts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh
        )

        n_inliers = int(np.sum(inliers)) if inliers is not None else 0
        return A23, n_matches, inliers, mkpts0, mkpts1
    else:
        return None, 0, np.array([], dtype=np.int8), None, None

def estimate_affine_with_rotations(img0, img1, extractor, matcher, ransac_thresh=3.0):
    # 对 img1 多旋转进行匹配估计仿射变换
    # 参数：
    #     img0: torch.Tensor [C,H,W]
    #     img1: torch.Tensor [C,H,W]
    #     extractor: SuperPoint 特征提取器
    #     matcher: LightGlue 匹配器
    # 返回：
    #     A23: 估计的仿射矩阵 (2×3)，若失败则为 None
    #     n_matches: 合并后的匹配点对数量
    #     n_inliers: RANSAC 内点数量，若失败则为 0
    with torch.no_grad():
        feats0 = extractor.extract(img0)
    return estimate_affine_with_rotations_feat(feats0, img1, extractor, matcher, ransac_thresh=3.0)

def estimate_affine(img0, img1, extractor, matcher, ransac_thresh=3.0):
    # 3. 提取 SuperPoint 特征
    with torch.no_grad():
        feats0 = extractor.extract(img0)
        feats1 = extractor.extract(img1)

    # 4. LightGlue 匹配
    with torch.no_grad():
        matches = matcher({'image0': feats0, 'image1': feats1})

    # 5. 提取匹配点对
    kpts0 = feats0['keypoints'][0].cpu().numpy()
    kpts1 = feats1['keypoints'][0].cpu().numpy()
    matches0 = matches['matches0'][0].cpu().numpy()
    valid = matches0 > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches0[valid]]

    n_matches = len(mkpts0)

    if n_matches > 2:
        # 6. 估计仿射变换 A23 (from img0 → img1)
        A23, inliers = cv2.estimateAffinePartial2D(mkpts0, mkpts1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        return A23, n_matches, inliers, mkpts0, mkpts1
    else:
        return None, n_matches, 0, mkpts0, mkpts1