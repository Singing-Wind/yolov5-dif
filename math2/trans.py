import numpy as np
import torch

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

def invA23_batch(A):
    # 求多个 2×3 仿射矩阵的逆。
    # 输入:
    #     A: np.ndarray of shape (b, 2, 3)
    # 返回:
    #     A_inv: np.ndarray of shape (b, 2, 3)
    is_tensor = isinstance(A, torch.Tensor)

    if is_tensor:
        R = A[:, :, :2]              # (b, 2, 2)
        t = A[:, :, 2]               # (b, 2)
        R_inv = torch.linalg.inv(R) # (b, 2, 2)
        t_inv = -torch.bmm(R_inv, t.unsqueeze(2)).squeeze(2)  # (b, 2)
        A_inv = torch.cat([R_inv, t_inv.unsqueeze(2)], dim=2)  # (b, 2, 3)
        return A_inv.to(dtype=torch.float32)
    
    else:  # numpy
        R = A[:, :, :2]
        t = A[:, :, 2]
        R_inv = np.linalg.inv(R)
        t_inv = -np.einsum('bij,bi->bj', R_inv, t)
        A_inv = np.concatenate([R_inv, t_inv[:, :, None]], axis=2)
        return A_inv.astype(np.float32)


def transform_points(xyxy, A23):
    """将xyxy（8个元素）按行拼成 [4,2]，添加1列做为齐次变换，进行逆变换"""
    pts = xyxy.reshape(4, 2)
    pts_h = torch.hstack([pts, torch.ones((4,1),device=pts.device)])  # [4,3]
    pts_trans = (A23 @ pts_h.T).T  # pts_trans[4,2]
    return pts_trans