import numpy as np

def to_homogeneous(A23):
    """将 2×3 仿射矩阵转为 3×3 齐次矩阵"""
    return np.vstack([A23, [0, 0, 1]])

def affine_cosine_similarity(A23_pred, A23_gt):
    # 整体评估 A23_pred 与 A23_gt 的相似程度
    # 返回值 sim 越接近 1 越相似
    A33p = to_homogeneous(A23_pred)
    A33g = to_homogeneous(A23_gt)

    dR33 = np.linalg.inv(A33p) @ A33g

    sim = (np.trace(dR33) - 1) / 2
    return sim