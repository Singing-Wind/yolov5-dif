import cv2
import numpy as np
import math
import random
import torch
import os
from collections import deque
import pygame
from pygame.locals import *
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Placeholder for the predict function
# from predict import predict
from math2.transform import A232rot_torch,xcycdudv2A23_torch,A23inverse,A23inverse_torch,apply_inverse_affine_and_map,A23Crop_torch,map_arrow
# from model import RemoteVitModel
from general.config import load_config
from image.tif import read_tif_with_tifffile
from FlyCrop_copy import FlyCrop
from image.draw_surface import draw_arrow
from models.experimental import attempt_load
from detect import detect
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii, non_max_suppression,non_max_suppression_obb, non_max_suppression_dfl, \
    apply_classifier, scale_coords,scale_coords_poly, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box,xyxyxyxy2xywhr
from tools.plotbox import plot_one_box,plot_one_rot_box

from copy import deepcopy
from models.yolo import OUT_LAYER

COLORS = [
        (54, 67, 244),
        (99, 30, 233),
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

def draw_map_with_uavs(thumbnail_surface, map_shape, q, crop_size, current_t, current_p, init=True):
    # 在黑色背景的缩略图上按比例绘制真实轨迹和预测轨迹，并添加边框和网格线。
    # 参数:
    #     map_image (np.ndarray): 原始大图像，形状为 [H, W, C]
    #     q (deque): 轨迹队列，包含(t, p)元组
    #     crop_size (tuple): 裁剪大小 (width, height)
    #     current_t (tuple): 当前的真实轨迹 (xc, yc, cos_t, sin_t)
    #     current_p (tuple): 当前的预测轨迹 (xcp, ycp, dup, dvp)
    #     thumbnail_size (tuple): 缩略图的大小 (width, height)
    # 返回:
    #     thumbnail_surface (pygame.Surface): Pygame表面对象，包含绘制的轨迹和网格
    thumbnail_width, thumbnail_height = thumbnail_surface.get_size()

    map_height, map_width = map_shape.shape[0], map_shape.shape[1]

    # 计算缩放因子，保持长宽比
    scale = min(thumbnail_width / map_width, thumbnail_height / map_height)

    # 计算偏移量以居中
    pad_x = (thumbnail_width - map_width * scale) / 2
    pad_y = (thumbnail_height - map_height * scale) / 2

    if init:
        thumbnail_surface.fill((0, 0, 0))  # 填充黑色背景
        # 计算网格间距，取crop_size的最大值，并按比例缩放
        grid_spacing = max(crop_size) * scale

        # 绘制网格线
        dark_gray = (50, 50, 50)  # 暗灰色

        # 绘制垂直网格线
        current_x = pad_x
        while current_x <= thumbnail_width - pad_x:
            pygame.draw.line(thumbnail_surface, dark_gray, (int(current_x), pad_y), (int(current_x), thumbnail_height - pad_y), 1)
            current_x += grid_spacing

        # 绘制水平网格线
        current_y = pad_y
        while current_y <= thumbnail_height - pad_y:
            pygame.draw.line(thumbnail_surface, dark_gray, (pad_x, int(current_y)), (thumbnail_width - pad_x, int(current_y)), 1)
            current_y += grid_spacing

        # 绘制整体边框，基于缩放和偏移
        border_color = (255, 255, 255)  # 白色
        pygame.draw.rect(
            thumbnail_surface,
            border_color,
            pygame.Rect(pad_x, pad_y, map_width * scale, map_height * scale),
            2
        )

    # 绘制轨迹线
    for i in range(1, len(q)):
        t_prev = q[i-1]
        t_curr = q[i]

        # 真实轨迹点
        x1, y1 = t_prev[0], t_prev[1]
        x2, y2 = t_curr[0], t_curr[1]

        # 缩放和偏移
        x1_scaled = x1 * scale + pad_x
        y1_scaled = y1 * scale + pad_y
        x2_scaled = x2 * scale + pad_x
        y2_scaled = y2 * scale + pad_y

        # 绘制真实轨迹线（红色）
        pygame.draw.line(
            thumbnail_surface,
            (255, 0, 0),
            (int(x1_scaled), int(y1_scaled)),
            (int(x2_scaled), int(y2_scaled)),
            2
        )


    # 绘制当前裁剪矩形
    if current_t[0] is not None and current_t[1] is not None:
        # xc, yc, cos_t, sin_t, A23t = current_t
        # # 缩放中心点
        # xc_scaled = xc * scale + pad_x
        # yc_scaled = yc * scale + pad_y
        # # 缩放裁剪框尺寸
        # w_scaled = crop_size[0] * scale
        # h_scaled = crop_size[1] * scale
        # # 创建裁剪框矩形
        # rect = pygame.Rect(0, 0, w_scaled, h_scaled)
        # rect.center = (int(xc_scaled), int(yc_scaled))
        # # 绘制裁剪框矩形
        # pygame.draw.rect(thumbnail_surface, (0, 255, 0), rect, 2)
        # 提取 A23t 和 A23p 矩阵并转换为 numpy 数组
        A23t = current_t[4].cpu().numpy()  # [2, 3] 矩阵
        # 计算逆仿射矩阵
        A23t_inv = A23inverse(A23t)

        # 定义裁剪矩形的四个顶点
        crop_corners = [
            (0, 0),
            (crop_size[0], 0),
            (crop_size[0], crop_size[1]),
            (0, crop_size[1])
        ]

        # 应用逆仿射变换到裁剪矩形顶点，得到 t 和 p 的四边形坐标，并映射到缩略图
        t_mapped = apply_inverse_affine_and_map(A23t_inv, crop_corners, scale, pad_x, pad_y)  # [ (x', y'), ... ]
        # 绘制四边形
        pygame.draw.polygon(thumbnail_surface, (255, 0, 0), t_mapped, 2)   # 绘制 t 的四边形，红色

        if current_p is not None:
            A23p = current_p[4].cpu().numpy()  # [2, 3] 矩阵
            A23p_inv = A23inverse(A23p)
            p_mapped = apply_inverse_affine_and_map(A23p_inv, crop_corners, scale, pad_x, pad_y)  # [ (x', y'), ... ]
            pygame.draw.polygon(thumbnail_surface, (0, 255, 0), p_mapped, 2)   # 绘制 p 的四边形，绿色

        #draw two poly arrows x-y
        # 定义裁剪矩形的中心点和方向向量
        #draw two poly arrows x-y 定义裁剪矩形的中心点和方向向量
        unit_up = np.array([0, -1]) # 图像朝上的单位矢量
        p_center_mapped,p_up_end,p_rot_end = map_arrow(crop_size,A23t_inv, unit_up, scale, 2.0, pad_x, pad_y)

        # 绘制方向轴
        draw_arrow(thumbnail_surface, (0, 255, 0), p_center_mapped, p_up_end, arrow_size=10)   # p 的朝上方向轴，绿色
        draw_arrow(thumbnail_surface, (255, 0, 0), p_center_mapped, p_rot_end, arrow_size=10)   # p 的顺时针旋转90度本体x方向轴，红色

    return thumbnail_surface

def draw_surface(surface, A23p,A23t, crop_size, scale, colorp=(255, 120, 0), colort=(255, 255, 0)):#color:BGR
    # 定义裁剪矩形的四个顶点
    crop_corners = [
        (0, 0),
        (crop_size[0], 0),
        (crop_size[0], crop_size[1]),
        (0, crop_size[1])
    ]
    surface.fill((0, 0, 0))  # 填充黑色背景
    # 计算逆仿射矩阵
    if A23p is not None:
        A23p_inv = A23inverse_torch(A23p).cpu().numpy()
        #center
        #draw two poly arrows x-y 定义裁剪矩形的中心点和方向向量
        p_center_mapped,p_up_end,p_rot_end = map_arrow(crop_size,A23p_inv, np.array([0, -1]), scale, 1.0)

        # 绘制方向轴
        draw_arrow(surface, (0, 255, 0), p_center_mapped, p_up_end, arrow_size=10)   # p 的朝上方向轴，绿色
        draw_arrow(surface, (255, 0, 0), p_center_mapped, p_rot_end, arrow_size=10)   # p 的顺时针旋转90度本体x方向轴，红色

        # 应用逆仿射变换到裁剪矩形顶点，得到 t 和 p 的四边形坐标，并映射到缩略图
        p_mapped = apply_inverse_affine_and_map(A23p_inv, crop_corners, scale, 0, 0)  # [ (x', y'), ... ]
        # 绘制四边形
        pygame.draw.polygon(surface, colorp, p_mapped, 2)   # 绘制 p 的四边形，绿色
    #
    # 计算逆仿射矩阵
    A23t_inv = A23inverse_torch(A23t)
    # 应用逆仿射变换到裁剪矩形顶点，得到 t 和 p 的四边形坐标，并映射到缩略图
    p_mapped = apply_inverse_affine_and_map(A23t_inv, crop_corners, scale, 0, 0)  # [ (x', y'), ... ]
    # 绘制四边形
    pygame.draw.polygon(surface, colort, p_mapped, 2)   # 绘制 p 的四边形，绿色

def draw_video_image(frame):
    if frame is None:
        return None
    video_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_rgb = cv2.resize(video_rgb, (512, 512))
    video_surface = pygame.surfarray.make_surface(video_rgb.swapaxes(0,1))
    return video_surface

def draw_3d_view(map_image, A23t, crop_size, scale, pad_x, pad_y):
    # 从大图中提取一个小块，并在该小块上绘制真值四边形。
    # 参数:
    #     map_image (np.ndarray): 原始大图像，形状为 [H, W, C]
    #     A23t (np.ndarray): 真值的 2x3 仿射变换矩阵
    #     crop_size (tuple): 裁剪大小 (width, height)
    #     scale (float): 缩放因子
    #     pad_x (float): 缩略图在 x 方向上的偏移量
    #     pad_y (float): 缩略图在 y 方向上的偏移量
    # 返回:
    #     map_surface (pygame.Surface): Pygame 表面对象，包含绘制的真值四边形
    import numpy as np
    import pygame
    import cv2

    # 计算逆仿射变换矩阵
    A23t_inv = A23inverse(A23t)

    # 使用逆仿射变换提取裁剪区域
    # 设置输出大小为 crop_size
    warp_size = (crop_size[0], crop_size[1])
    cropped_patch = cv2.warpAffine(map_image, A23t_inv, warp_size, flags=cv2.INTER_LINEAR)

    # Convert to Pygame surface
    # OpenCV uses BGR, convert to RGB
    cropped_patch_rgb = cv2.cvtColor(cropped_patch, cv2.COLOR_BGR2RGB)
    # 转换为 Pygame Surface
    map_surface = pygame.surfarray.make_surface(cropped_patch_rgb.swapaxes(0,1))

    # 绘制真值四边形（已提取裁剪区域，对应整个区域）
    pygame.draw.rect(map_surface, (255, 0, 0), pygame.Rect(0, 0, crop_size[0], crop_size[1]), 2)

    return map_surface

def pts2cov(pts):  # n * 8
    """
    pts -> (cx, cy, a, b, c)
    """
    n = pts.shape[0]
    if n:
        # 目标自身的宽高
        cx = pts[:, ::2].sum(dim=1) / 4
        cy = pts[:, 1::2].sum(dim=1) / 4
        xf, yf = (pts[:, 0] + pts[:, 2]) / 2, (pts[:, 1] + pts[:, 3]) / 2
        xb, yb = (pts[:, 4] + pts[:, 6]) / 2, (pts[:, 5] + pts[:, 7]) / 2
        dx = xf - xb
        dy = yf - yb
        L2 = dx ** 2 + dy ** 2
        L = torch.sqrt(L2)
        # 方向
        cos = dx / L
        sin = dy / L

        # cx = (pts[:, 0] + pts[:, 2] + pts[:, 4] + pts[:, 6]) / 4
        # cy = (pts[:, 1] + pts[:, 3] + pts[:, 5] + pts[:, 7]) / 4
        txy = torch.zeros_like(pts, dtype=torch.float)
        px = pts[:, ::2] - cx[:, None]
        py = pts[:, 1::2] - cy[:, None]
        txy[:, ::2] = cos[:, None] * px + sin[:, None] * py
        txy[:, 1::2] = -sin[:, None] * px + cos[:, None] * py
        lx = (txy[:, 4] + txy[:, 6]) / 2
        rx = (txy[:, 0] + txy[:, 2]) / 2
        a = (rx - lx) / 2
        ty = (txy[:, 1] + txy[:, 7]) / 2
        by = (txy[:, 3] + txy[:, 5]) / 2
        b = (by - ty) / 2
        # assert any(a) > 0
        # assert any(b) > 0
        
        cos2 = cos.pow(2)
        sin2 = sin.pow(2)
        a, b = a ** 2 / 12, b ** 2 / 12
        a, b, c = a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin
        out = torch.stack((cx, cy, a, b, c), dim=1)
    else:
        out = torch.zeros(0, 5, dtype=torch.float32).to(pts.device)
    return out

def probiou_fly(obb1, obb2, eps=1e-7):
    x1, y1, a1, b1, c1 = obb1.split(1, dim=-1)
    x2, y2, a2, b2, c2 = (x.squeeze(-1)[None] for x in obb2.split(1, dim=-1))
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    return iou

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
if __name__ == '__main__':
    config_path = 'fly/config-DOTA.json'
    config = load_config(config_path)
    # Simulation parameters
    CROP_SIZE = (512, 512)
    OBJECT_LIMIT = 256
    QUEUE_LIMIT = 8

    # Initialize FlyCrop
    images_dir=config['images_dir']
    image_names = [f for f in os.listdir(images_dir) if f.endswith(('.tif','.png')) and f[0]!='#']
    conf_thres = config.get('conf_thres', 0.25)
    iou_thres = config.get('iou_thres', 0.45)
    fly_num = max(config.get('fly_num', 4), 1)
    #images = read_tif_with_tifffile(os.path.join(images_dir,image_names[0]))
    index = 0
    fly = FlyCrop()

    flys = [fly] + [FlyCrop(fly_base=fly) for _ in range(fly_num - 1)]
    mul_n = np.ceil(np.sqrt(fly_num))
    EMPTY_SIZE = int(CROP_SIZE[0] * mul_n), int(CROP_SIZE[1] * mul_n), 3
    [fly.set_size(CROP_SIZE) for fly in flys]
    flys[0].reload(os.path.join(images_dir,image_names[0]))  # Replace with your large map image path

    flys_q = [deque(maxlen=QUEUE_LIMIT) for _ in range(fly_num)]

    # Initialize queue
    globe_objects = torch.zeros(0, 10)
    labels_cov = torch.zeros(0, 5)

    # Initialize Pygame
    pygame.init()
    window_width, window_height = 1200, 800
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Real-Time FlyCrop Simulation")

    # Font for displaying text
    font = pygame.font.SysFont(None, 24)
    [fly.init_params() for fly in flys]
    # Placeholder model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    globe_objects = globe_objects.to(device)
    labels_cov = labels_cov.to(device)
    #must keep same with train.py
    # model = RemoteVitModel(vit_head_out_num=1024, dudv_size=32, hidden_size_xcyc=0, hidden_size_rrotation=128,C=256).to(device)
    if config['model_best_path'] and os.path.exists(config['model_best_path']):
        model = attempt_load(config['model_best_path'], map_location=device)  # load FP32 model
        # ckpt = torch.load(config['model_best_path'], weights_only=False)
        # model.load_state_dict(ckpt['model_state_dict'])
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names

        for mname in OUT_LAYER.keys():
            m = model.get_module_byname(mname)
            if m is not None:
                break
        mname = OUT_LAYER[mname]
        model.eval()

        # Main loop variables
        frame_count = 0

        # Main Loop
        running = True
        clock = pygame.time.Clock()

        # 创建一个黑色的Pygame Surface
        thumbnail_size=(600, 600)
        thumbnail_surface = pygame.Surface((thumbnail_size[0], thumbnail_size[1]))
        Focus_size=(300, 300)
        Focus_surface = pygame.Surface((Focus_size[0], Focus_size[1]))
        show_3d = 0

        thumbnail_width, thumbnail_height = thumbnail_surface.get_size()
        map_height, map_width = flys[0].map[0].shape[0], flys[0].map[0].shape[1]
        # 计算缩放因子，保持长宽比
        scale = min(thumbnail_width / map_width, thumbnail_height / map_height)
        # 计算偏移量以居中
        pad_x = (thumbnail_width - map_width * scale) / 2
        pad_y = (thumbnail_height - map_height * scale) / 2

        while running and frame_count < 99999999:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        index = (index + 1) % len(image_names)
                        flys[0].reload(os.path.join(images_dir,image_names[index]))  # Replace with your large map image path
                        [q.clear() for q in flys_q]
                        [fly.init_params() for fly in flys]
                        thumbnail_width, thumbnail_height = thumbnail_surface.get_size()
                        map_height, map_width = flys[0].map[0].shape[0], flys[0].map[0].shape[1]
                        # 计算缩放因子，保持长宽比
                        scale = min(thumbnail_width / map_width, thumbnail_height / map_height)
                        # 计算偏移量以居中
                        pad_x = (thumbnail_width - map_width * scale) / 2
                        pad_y = (thumbnail_height - map_height * scale) / 2
            
            # Update FlyCrop
            image = None
            A23t, t = [], []
            image_shows = np.zeros(EMPTY_SIZE, dtype=np.uint8)
            for idx, fly in enumerate(flys):
                fly.get_affine()
                fly.bump()
                image = fly.render() if image is None else torch.cat([image, fly.render()], dim=0) #image[B,C,H,W]
            
                # Handle yaw rotation
                if random.uniform(0, 1) < 0.01:
                    dth = math.pi * random.uniform(-10, 10) / 180
                    fly.rot(dth)
                
                # Update speed
                if random.uniform(0, 1) < 0.02:
                    fly.update_speed(0.2)
                
                # Motion
                fly.motion()
                
                # Convert A23 to t
                A23t.append(fly.mA.squeeze(0))
                t_cos, t_sin, t_s, t_xc, t_yc = A232rot_torch(A23t[-1], CROP_SIZE)
                t.append((t_xc, t_yc, t_cos, t_sin, A23t[-1]))


            # Prepare image batch for prediction (dummy)
            if image is not None:
                # with torch.no_grad():  # 确保不跟踪梯度
                #     objp, xcp, ycp, dup, dvp = predict(model, image / 255.0) #[1]
                # xcp*=fly.map_t.shape[-1]
                # ycp*=fly.map_t.shape[-2]
                # A23p = xcycdudv2A23_torch(xcp, ycp, dup, dvp,CROP_SIZE)
                # p = (xcp[0], ycp[0], dup[0], dvp[0], A23p)
                img = image.clone().to(device)[:, [2,1,0]].contiguous()
                img = img / 255.0  # 0 - 255 to 0.0 - 1.0
                #img[c=3,h,w]
                if len(img.shape) == 3:
                    img = img[None]  # expand for batch dim
                #img[1,3,H,W]
                # Inference
                pred = detect(model, img, False, conf_thres, iou_thres, mname=mname,agnostic_nms=False,classes=None,max_det=3000)
                #
                # Process predictions
                
                for idx, det in enumerate(pred):  # per image batch循环
                    image_show = image[idx].permute(1, 2, 0).contiguous().to(device='cpu').numpy()#.astype(np.uint8) #image[1,C,H,W]->[C,H,W]->image_show[H,W,C]
                    if len(det): #det[np,4(pts)*2+1(conf)+1(cls)]
                        # Rescale boxes from img_size to image_show size
                        det[:, :8] = scale_coords_poly(img.shape[2:], det[:, :8], image_show.shape).round()
                        # Print results
                        s = ''
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            xyxy = [x.cpu() for x in xyxy]
                            label = '{} {:.2f}'.format(names[int(cls)], conf)
                            dir_line = 1
                            line_thickness = 2
                            if mname in [0, 1]:
                                plot_one_box(np.array(xyxy), image_show, color=COLORS[int(cls)%len(COLORS)], label=label)
                            else:
                                plot_one_rot_box(np.array(xyxy), image_show, color=COLORS[int(cls)%len(COLORS)], label=label, dir_line=dir_line, line_thickness=line_thickness)
                        print(s)

                        det_new = torch.zeros((det.shape[0], 10), device=device, dtype=det.dtype) #det_new[np,4(pts)*2+1(conf)+1(cls)]
                        A23_inv = A23inverse_torch(t[idx][4]) #(t_xc, t_yc, t_cos, t_sin, A23t[-1])
                        if mname in [0, 1]:
                            det_pts = det[:, [0, 1, 2, 1, 2, 3, 0, 3]].view(-1, 4, 2)   # x1y1 x2y1, x2y2, x1y2
                        else:
                            # 计算逆仿射矩阵
                            det_pts = det[:, :8].view(-1, 4, 2)
                        det_pts = torch.cat([det_pts, torch.ones_like(det_pts[:, :, :1])], dim=-1)
                        det_pts = torch.einsum('ijk, km->ijm', det_pts, A23_inv.T)
                        det_new[:, :8] = det_pts.view(-1, 8)
                        det_new[:, -2:] = det[:, -2:]
                        det = det_new
                        det_cov = pts2cov(det)  # 中心 + 协方差参数
                        if globe_objects.shape[0] > 0:
                            iou_ = probiou_fly(det_cov, labels_cov) > iou_thres
                            cls_ = det[:, -1:] == globe_objects[:, -1]
                            res = iou_ & cls_
                            conf = det[:, -2:-1] < globe_objects[:, -2] * res
                            max_conf = torch.argmax(conf * globe_objects[:, -2:-1].T, dim=1)
                            max_conf = max_conf[conf.any(-1)]    # 容器内置信度 大于 新标签的
                            det[conf.any(-1)] = globe_objects[max_conf]  # 提取容器内大于新标签的
                            det_cov[conf.any(-1)] = labels_cov[max_conf]  # 提取容器内大于新标签的
                            res = res.any(0)
                            globe_objects = globe_objects[~res]
                            labels_cov = labels_cov[~res]
                        globe_objects = torch.cat([det, globe_objects], dim=0)[:OBJECT_LIMIT]
                        labels_cov = torch.cat([det_cov, labels_cov], dim=0)[:OBJECT_LIMIT]
                    h_0 = int(idx // mul_n * CROP_SIZE[0])
                    h_1 = h_0 + CROP_SIZE[0]
                    w_0 = int(idx % mul_n * CROP_SIZE[1])
                    w_1 = w_0 + CROP_SIZE[1]
                    image_shows[h_0:h_1, w_0:w_1, :] = image_show
                    # Print time (inference-only)
                    # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            
            # Append to queue
            [q.append(deepcopy(t[idx])) for idx, q in enumerate(flys_q)]
            
            # Visualization
            screen.fill((0, 0, 0))  # Clear screen
            
            # Left Side
            # Top: image
            video_surface = draw_video_image(cv2.resize(image_shows, (CROP_SIZE[1], CROP_SIZE[0])))
            if video_surface:
                screen.blit(video_surface, (0, 0))
            
            # Bottom: map with trajectories
            for idx, fly in enumerate(flys):
                thumbnail_surface = draw_map_with_uavs(thumbnail_surface, fly.map[0], flys_q[idx], CROP_SIZE, t[idx], None, idx==0)
                A23p_inv = np.asarray([[1, 0, 0], [0, 1, 0]])
                for l in globe_objects:
                    xyxy = l[:8].view(4, 2).cpu().numpy()
                    cls = l[-1]
                    p_mapped = apply_inverse_affine_and_map(A23p_inv, xyxy, scale, pad_x, pad_y)  # [ (x', y'), ... ]
                    color = COLORS[int(cls)%len(COLORS)]
                    color = color[2], color[1], color[0]
                    pygame.draw.polygon(thumbnail_surface, color, p_mapped, 2)   # 绘制 p 的四边形，绿色
            screen.blit(thumbnail_surface, (600, 0))

            #
            big_crop_size = (1024,1024)
            A23Tct,A23padt = A23Crop_torch(A23t[idx],CROP_SIZE,big_crop_size)
            # A23Tcp,A23padp = A23Crop_torch(A23p,CROP_SIZE,big_crop_size)
            # pad_croped_image = cv2.warpAffine(fly.map, A23Tct.numpy(), big_crop_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            # pad_croped_image_rgb = cv2.cvtColor(pad_croped_image, cv2.COLOR_BGR2RGB)
            # pad_croped_image_rgb = pygame.surfarray.make_surface(pad_croped_image_rgb)
            # # 3. 渲染到 Focus_surface 上
            # Focus_surface.blit(pad_croped_image_rgb, (0, 0))
            draw_surface(Focus_surface, None, A23padt, CROP_SIZE, Focus_size[0] / big_crop_size[0], colorp=(255, 128, 0), colort=(255, 255, 0)) #yellow: gt
            screen.blit(Focus_surface, (0, 512))
            
            # Right Side
            # 3D view simulation
            if show_3d:
                cos_t, sin_t, s, xc, yc = t_cos, t_sin, t_s, t_xc, t_yc
                map_3d_surface = draw_3d_view(fly.map, xc, yc, CROP_SIZE)
                screen.blit(map_3d_surface, (0, 512))
            
            # Update the display
            pygame.display.flip()
            
            # Limit to 30 FPS
            clock.tick(30)
            frame_count += 1

    pygame.quit()
    sys.exit()
