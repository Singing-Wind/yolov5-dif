import pygame
import pygame.gfxdraw
import math
import numpy as np
import torch
import cv2


# 绘制坐标轴、刻度
def draw_axes(surface,margin_left,margin_bottom):
    axis_color = (0, 0, 0)
    width = surface.get_width()
    plot_width = 0.95*width
    height = surface.get_height()
    plot_height = 0.95*height
    
    font = pygame.font.SysFont('Arial', 14)
    
    # Y轴
    pygame.draw.line(surface, axis_color, (margin_left, 20), (margin_left, height - margin_bottom), 2)
    # X轴
    pygame.draw.line(surface, axis_color, (margin_left, height - margin_bottom), (width - 20, height - margin_bottom), 2)

    # Y轴刻度
    for i in range(6):  # 0.0 到 1.0，每隔 0.2
        val = i * 0.2
        y = 20 + plot_height * (1 - val)
        pygame.draw.line(surface, axis_color, (margin_left - 5, int(y)), (margin_left + 5, int(y)), 1)
        label = font.render(f'{val:.1f}', True, axis_color)
        surface.blit(label, (margin_left - 40, int(y) - 7))

    # X轴刻度
    for i in range(0, T, T // 10):  # 每 10%
        x = margin_left + int(i * plot_width / (T - 1))
        pygame.draw.line(surface, axis_color, (x, height - margin_bottom - 5), (x, height - margin_bottom + 5), 1)
        label = font.render(f'{i}', True, axis_color)
        surface.blit(label, (x - 10, height - margin_bottom + 8))

MAX_ELLIPSE_PTS = 20
def ellipse_LUT(num_points):
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    return np.cos(t), np.sin(t)
cos_LUT, sin_LUT = ellipse_LUT(MAX_ELLIPSE_PTS)

def draw_rotated_ellipse(surface, color, xyabcs, filled=1, num_points=MAX_ELLIPSE_PTS, mult=1.0):
    cx, cy = xyabcs[:2]
    a, b = xyabcs[2:4]
    cs = xyabcs[4:6]

    points = []
    for i in range(num_points):
        t = 2 * math.pi * i / num_points
        x = mult * a * cos_LUT[i]
        y = mult * b * sin_LUT[i]
        # 旋转
        xr = x * cs[0] - y * cs[1]
        yr = x * cs[1] + y * cs[0]
        points.append((int(cx + xr), int(cy + yr)))

    if filled == 1:
        pygame.gfxdraw.filled_polygon(surface, points, color)
    else:
        pygame.gfxdraw.aapolygon(surface, points, color)

def obb2gauss_hot(obb, scale=0.75):
    """
    将旋转框 [x, y, w, h, cosθ, sinθ] 转换为高斯分布参数
    scale 控制 std 的大小（比例越小，分布越“集中”）
    """
    xy = obb[:, 0:2]               # (B, 2)
    wh = obb[:, 2:4] * scale       # 缩小后再平方得到 σ²
    cs = obb[:, 4:6]               # (cosθ, sinθ)
    
    cosθ, sinθ = cs[:, 0], cs[:, 1]

    # 旋转矩阵 R
    R = torch.stack([
        torch.stack([cosθ, -sinθ], dim=-1),
        torch.stack([sinθ,  cosθ], dim=-1)
    ], dim=-2)                    # (B, 2, 2)

    # 对角矩阵 D = diag(σx², σy²)
    sx2 = (wh[:, 0]) ** 2
    sy2 = (wh[:, 1]) ** 2

    D = torch.stack([
        torch.stack([sx2, torch.zeros_like(sx2)], dim=-1),
        torch.stack([torch.zeros_like(sy2), sy2], dim=-1)
    ], dim=-2)                   # (B, 2, 2)

    # 协方差矩阵 Σ = R D R^T
    Sigma = R @ D @ R.transpose(-2, -1)  # (B, 2, 2)
    return xy, Sigma

def batched_multivariate_normal_pdf(xy_grid, xyabcsC):
    """
    计算多个二维高斯分布在网格上的概率密度
    
    Args:
        xy_grid: (H, W, 2) 的网格坐标
        mu:      (B, 2) 的均值向量, B 是分布数量
        Sigma:   (B, 2, 2) 的协方差矩阵
    Returns:
        (B, H, W) 的概率密度图
    """
    B = xyabcsC.shape[0]  # 分布数量
    # xyabc = obb2gauss(xyabcsC[:, :-1])
    # center = xyabc[:, :2]
    # Sigma = xyabc[:, [2, 4, 4, 3]].view(-1, 2, 2)
    center, Sigma = obb2gauss_hot(xyabcsC[:, :-1], 2.25)

    # center = torch.tensor([55, 105]).float().to('cuda').view(-1, 2)
    # Sigma = torch.tensor([50, 30, 30, 50]).float().to('cuda').view(-1, 2, 2)
    n = center.shape[-1]  # 维度（此处为2）
    
    # 调整维度以支持广播: (B, H, W, 2)
    diff = xy_grid.unsqueeze(0) - center.reshape(B, 1, 1, 2)
    
    # 批量计算逆矩阵和行列式 (B, 2, 2)
    inv_Sigma = torch.linalg.inv(Sigma)
    det_Sigma = torch.det(Sigma)
    
    # 计算指数部分: (B, H, W)
    exponent = -0.5 * torch.einsum('bhwi,bij,bhwj->bhw', diff.half(), inv_Sigma.half(), diff.half())
    
    # 计算归一化系数 (B,)
    norm = 1.0 / ((2 * torch.pi) ** (n/2) * torch.sqrt(det_Sigma)).reshape(B, 1, 1)

    norm = norm * torch.exp(exponent)
    n_min, n_max = norm.amin((1, 2)).view(-1, 1, 1), norm.amax((1, 2)).view(-1, 1, 1)
    norm = ((norm - n_min) / (n_max - n_min + 1e-8) * (1 + -2 * xyabcsC[:, -1].view(-1, 1, 1)))#.sum(0, keepdims=False).clamp_(-1, 1)
    
    # norm_img = torch.ones_like(norm) * 0.5
    # norm_imgs = (torch.stack([norm_img + norm * 0.5, norm_img, norm_img - norm * 0.5], dim=-1) * 255).to(torch.uint8).cpu().numpy()
    # if thumbnail_size is not None:
    #     norm_imgs = cv2.resize(norm_imgs, thumbnail_size)
    # norm_surface = pygame.surfarray.make_surface(norm_imgs.swapaxes(0,1))
    return norm

def draw_oriented_fan(xyabcsC, globe_surface):
    x, y, a, b, cos_theta, sin_theta, team = xyabcsC.split(1, dim=-1)
    theta = torch.atan2(sin_theta, cos_theta)
    radiusS = (torch.maximum(a, b) * 5).long()
    angle_spanS = torch.atan2(a, b) * 2  # 扇形弧度

    for idx in range(xyabcsC.shape[0]):
        if globe_surface[idx] is None:
            theta[idx] = theta[idx] if a[idx] > b[idx] else theta[idx] + torch.pi / 2
            radius = radiusS[idx]
            angle_span = angle_spanS[idx]
            fan_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            center = (radius, radius)
            color = (255, 0, 0) if team[idx].bool() else (0, 0, 255)
            # 扇形采样角度点
            num_points = 10
            angle_step = angle_span / num_points
            start_angle = -angle_span / 2

            for r_s in reversed(list(range(5))):  # 每隔几个像素一层，效率更高
                r = radius * (r_s + 1) / 5
                alpha = max(0, int(255 * (1 - r / radius)))
                points = [center]
                for i in range(num_points + 1):
                    angle = start_angle + i * angle_step
                    px = center[0] + r * math.cos(angle)
                    py = center[1] + r * math.sin(angle)
                    points.append((px.item(), py.item()))
                pygame.draw.polygon(fan_surface, (*color, alpha), points)

            # 旋转扇形并画到 screen 上
            rotated_fan = pygame.transform.rotate(fan_surface, -math.degrees(theta[idx].item()))
            fan_rect = rotated_fan.get_rect(center=(x[idx].item(), y[idx].item()))
            globe_surface[idx] = rotated_fan, fan_rect