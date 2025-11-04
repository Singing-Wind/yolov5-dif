import numpy as np
import math
import torch

def dudv2thetar(du,dv):
    s = torch.sqrt(du**2 + dv**2)
    cos_t,sin_t = du/s, dv/s
    return cos_t,sin_t,s

def xcycdudv2A23(xc,yc,du,dv,crop_size):#du=s*cos dv=s*sin
    w, h = crop_size
    A23 = np.array([[ du, dv, w/2 - ( xc*du + yc*dv)],
                    [-dv, du, h/2 - (-xc*dv + yc*du)]])
    return A23

def xcycdudv2A23_torch(xc,yc,du,dv,crop_size):#du=s*cos dv=s*sin
    w, h = crop_size
    A23 = torch.stack([du, dv, w/2 - ( xc*du + yc*dv), -dv, du, h/2 - (-xc*dv + yc*du)], dim=-1).view(-1, 2, 3)
    return A23

def A232rot(A23,crop_size=None):
    a11,a12 = A23[0,0],A23[0,1]
    s = math.sqrt(a11*a11+a12*a12)
    cos_t,sin_t = a11/s, a12/s
    if crop_size!=None:
        w, h = crop_size
        tx = -(A23[0,2] - w/2) / s #== xc*cos+yc*sin
        ty = -(A23[1,2] - h/2) / s #==-xc*sin+yc*cos
        xc = cos_t*tx-sin_t*ty
        yc = sin_t*tx+cos_t*ty
    else:
        xc,yc=None,None
    return cos_t,sin_t, s, xc,yc
def A232rot_torch(A23, crop_size=None):
    # 将 2x3 仿射变换矩阵转换为旋转向量、比例因子和中心点坐标。
    # 参数:
    #     A23 (torch.Tensor): 仿射变换矩阵，形状为 [2, 3] 或 [B, 2, 3]。
    #     crop_size (tuple, 可选): 裁剪区域大小 (宽, 高)。
    # 返回:
    #     cos_t (torch.Tensor): 旋转角的余弦值。
    #     sin_t (torch.Tensor): 旋转角的正弦值。
    #     s (torch.Tensor): 缩放因子。
    #     xc (torch.Tensor or None): 中心点的 x 坐标。
    #     yc (torch.Tensor or None): 中心点的 y 坐标。
    # 确保 A23 是浮点类型的张量
    A23 = A23.float()

    # 提取仿射矩阵的前两列元素
    a11, a12 = A23[..., 0, 0], A23[..., 0, 1]  # 支持批量输入 [B, 2, 3]

    # 计算缩放因子 s = sqrt(a11^2 + a12^2)
    s = torch.sqrt(a11 ** 2 + a12 ** 2)

    # 计算旋转角的余弦和正弦
    cos_t = a11 / s
    sin_t = a12 / s

    if crop_size is not None:
        w, h = crop_size

        # 计算 tx 和 ty
        tx = -(A23[..., 0, 2] - w / 2) / s
        ty = -(A23[..., 1, 2] - h / 2) / s

        # 计算中心点坐标 xc 和 yc
        xc = cos_t * tx - sin_t * ty
        yc = sin_t * tx + cos_t * ty
    else:
        xc, yc = None, None

    return cos_t, sin_t, s, xc, yc

def A23Crop(A23,crop_size,big_crop_size):
    cos_t,sin_t, s, xc,yc = A232rot(A23,crop_size)
    lt = (xc - big_crop_size[0] / 2, yc - big_crop_size[1] / 2)
    A23pad = A23.copy()
    A23pad[0,2] += s*( lt[0]*cos_t+lt[1]*sin_t)
    A23pad[1,2] += s*(-lt[0]*sin_t+lt[1]*cos_t)
    #
    A23Tc = np.array([[1, 0,-lt[0]],
                      [0, 1,-lt[1]]])
    return A23Tc, A23pad
def A23Crop_torch(A23,crop_size,big_crop_size):
    cos_t,sin_t, s, xc,yc = A232rot(A23,crop_size)
    lt = (xc - big_crop_size[0] / 2, yc - big_crop_size[1] / 2)
    A23pad = A23.clone()
    A23pad[0,2] += s*( lt[0]*cos_t+lt[1]*sin_t)
    A23pad[1,2] += s*(-lt[0]*sin_t+lt[1]*cos_t)
    #
    A23Tc = torch.tensor([[1, 0,-lt[0]],
                          [0, 1,-lt[1]]])
    return A23Tc, A23pad

def A23T(A23,At):
    cos_t,sin_t, s, _,_ = A232rot(A23)
    lt = (-At[0,2], -At[1,2])
    A23pad = A23.copy()
    A23pad[0,2] += s*( lt[0]*cos_t+lt[1]*sin_t)
    A23pad[1,2] += s*(-lt[0]*sin_t+lt[1]*cos_t)
    return A23pad

def A23inverse(A23):
    # 计算 A23 的逆阵
    A33 = np.vstack([A23, [0, 0, 1]])
    # 计算 A33 的逆
    A33_1 = np.linalg.inv(A33)
    # 取 A33_1 的前两行，得到 A23_1
    A23_1 = A33_1[:2, :]
    #A23_1 = np.linalg.pinv(A23).T
    return A23_1
def A23inverse_torch(A23):
    # 使用 PyTorch 计算 2x3 仿射变换矩阵 A23 的逆矩阵。
    # 参数:
    #     A23 (torch.Tensor): 形状为 (2, 3) 的张量，表示仿射变换矩阵。
    # 返回:
    #     A23_inv (torch.Tensor): 形状为 (2, 3) 的张量，表示逆仿射变换矩阵。
    # 确保 A23 是浮点型张量
    A23 = A23.float()

    # 创建 [0, 0, 1] 并拼接到 A23 下方，形成 3x3 矩阵
    ones = torch.tensor([[0, 0, 1]], dtype=A23.dtype, device=A23.device)
    A33 = torch.cat([A23, ones], dim=0)  # 形状: (3, 3)

    # 计算 A33 的逆
    A33_inv = torch.inverse(A33)  # 形状: (3, 3)

    # 提取逆矩阵的前两行，得到 A23_inv
    A23_inv = A33_inv[:2, :]  # 形状: (2, 3)

    return A23_inv

def A23_to_mapped_corners(A23, crop_size):
    # 获取裁切区域的四个顶点
    w,h = crop_size
    original_corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float32)
    
    # 计算 A23 的逆阵
    A23_1 = A23inverse(A23)
    
    # 使用逆矩阵映射四个顶点
    mapped_corners = original_corners@A23_1.T
      
    # 绘制从裁切区中心出发的两个正交的方向矢量
    # xc, yc, du, dv = label[1:].tolist()
    cos_t,sin_t, s, xc,yc = A232rot(A23,crop_size)
    # du,dv = s*cos_t, s*sin_t
    # height, width = big_image_shape[:2]
    center = np.array([xc, yc], dtype=np.float32)
    
    # 计算两个正交的单位方向矢量
    # L = math.sqrt(du*du+dv*dv)
    # vx1,vy1 = du/L, dv/L
    # t = np.arctan2(dv, du)
    unit_vec_1 = np.array([ cos_t, sin_t])  # 红色矢量
    unit_vec_2 = np.array([-sin_t, cos_t])  # 绿色矢量

    return mapped_corners,center,unit_vec_1,unit_vec_2, s

def apply_inverse_affine_and_map(A23_inv, points, scale, pad_x, pad_y):
    # 应用逆仿射变换矩阵 A23_inv 到一组点，并映射到缩略图坐标系。 
    # 参数:
    #     A23_inv (np.ndarray): 逆仿射变换矩阵，形状为 (2, 3)
    #     points (list of tuples): 要变换的点列表，格式为 [(x1, y1), (x2, y2), ...]
    #     scale (float): 缩放因子
    #     pad_x (float): 缩略图在 x 方向上的偏移量
    #     pad_y (float): 缩略图在 y 方向上的偏移量
    # 返回:
    #     mapped_points (list of tuples): 缩略图上的点列表，格式为 [(x1', y1'), (x2', y2'), ...]
    transformed_points = []
    for (x, y) in points:
        # 应用逆仿射变换
        x_prime = A23_inv[0, 0] * x + A23_inv[0, 1] * y + A23_inv[0, 2]
        y_prime = A23_inv[1, 0] * x + A23_inv[1, 1] * y + A23_inv[1, 2]
        # 缩放和偏移
        x_mapped = x_prime * scale + pad_x
        y_mapped = y_prime * scale + pad_y
        transformed_points.append((int(x_mapped), int(y_mapped)))
    return transformed_points

def map_arrow(crop_size,A23p_inv,unit_up, show_scale,arrow_scale, pad_x=0, pad_y=0):
    # unit_up = np.array([0, -1])  # 图像朝上的单位矢量
    center = (crop_size[0] / 2, crop_size[1] / 2)
    p_center_mapped = apply_inverse_affine_and_map(A23p_inv, [center], show_scale, pad_x, pad_y)[0]
    # 变换单位矢量 [0, -1]，得到 [vx, vy]
    transformed_v = A23p_inv[:2, :2].dot(unit_up)
    # 缩放方向矢量：乘以裁剪图像的高度和缩放因子
    transformed_v_scaled = transformed_v * crop_size[1]/2 * arrow_scale * show_scale
    # 终点坐标
    p_up_end = (p_center_mapped[0] + transformed_v_scaled[0], p_center_mapped[1] + transformed_v_scaled[1])

    # 计算顺时针旋转90度后的矢量 [-vy, vx]
    rotated_v = np.array([-transformed_v[1], transformed_v[0]])
    # 缩放方向矢量：乘以裁剪图像的宽度和缩放因子
    rotated_v_scaled = rotated_v * crop_size[0]/2 * arrow_scale * show_scale
    # 终点坐标
    p_rot_end = (p_center_mapped[0] + rotated_v_scaled[0], p_center_mapped[1] + rotated_v_scaled[1])
    return p_center_mapped,p_up_end,p_rot_end

# Utility functions
def cp_normalize(v, length):
    # 假设 v 是 torch.Tensor
    norm = torch.norm(v[:length])
    if norm.item() == 0:  # norm是一个标量张量，用.item()获取其数值
        return
    v[:length] = v[:length] / norm