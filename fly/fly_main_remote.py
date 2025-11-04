import cv2
import numpy as np
import math
import random
import torch
import os
from collections import deque
import pygame
from pygame.locals import *
import pygame_menu
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Placeholder for the predict function

from fly.math2.transform import A232rot_torch,xcycdudv2A23_torch,A23inverse,A23inverse_torch,apply_inverse_affine_and_map,A23Crop_torch,map_arrow
from general.config import load_config
from image.tif import read_tif_with_tifffile
from FlyCrop import FlyCrop,reload
from image.draw_surface import draw_arrow
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii, non_max_suppression,non_max_suppression_obb, non_max_suppression_dfl, \
    apply_classifier, scale_coords,scale_coords_poly, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box,xyxyxyxy2xywhr, \
    xywh2xyxy
from models.experimental import attempt_load
from detect import detect
from tools.plotbox import plot_one_box,plot_one_rot_box

from copy import deepcopy
from models.yolo import OUT_LAYER
from datetime import datetime
from pts2cov import pts2cov,probiou_fly_gauss,cs2arcarrow2

from surface import draw_axes,draw_rotated_ellipse, batched_multivariate_normal_pdf, draw_oriented_fan
from pathlib import Path
from fly.remote.model import RemoteVitModel
from fly.remote.image_crop import load_A23_file

from utils.metrics import ap_per_class, process_batch_obb

from general.MyString import replace_last_path

DATABAST_FLAG = False
database = None

if DATABAST_FLAG:
    from fly.database import DATABASE
    database = DATABASE()

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
    

def load_labels(path):
    with open(path, 'r') as fr:
        txt = np.array([x.split() for x in fr.read().strip().splitlines() if len(x)], dtype=np.float32).reshape(-1, 5)
    if txt.shape[0] > 0:
        if path.with_suffix('.pts').exists():
            with open(path.with_suffix('.pts'), 'r') as fr:
                pts = np.array([x.split() for x in fr.read().strip().splitlines() if len(x)], dtype=np.float32).reshape(-1, 8)
        else:
            xyxy = xywh2xyxy(txt[:, 1:])
            pts = np.stack([xyxy[:, 0], xyxy[:, 1], 
                            xyxy[:, 2], xyxy[:, 1], 
                            xyxy[:, 2], xyxy[:, 3], 
                            xyxy[:, 0], xyxy[:, 3], ], axis=-1, dtype=np.float32)
        labels = np.concatenate([txt[:, :1], pts], axis=-1, dtype=np.float32)
    else:
        labels = np.zeros((0, 9), dtype=np.float32)
    return labels

def swap_rb(color):
    # 交换一个 RGB 颜色元组的 R 和 B 分量
    # 参数:color (tuple): 一个 RGB 元组，如 (R, G, B)
    # 返回:tuple: 交换 R 和 B 后的新 RGB 元组，如 (B, G, R)
    r, g, b = color
    return (b, g, r)

def draw_map_with_uavs(thumbnail_surface, map_shape, q, crop_size, current_t, current_p, idx):
    # 在黑色背景的缩略图上按比例绘制真实轨迹和预测轨迹，并添加边框和网格线。
    # 参数:
    #     map_image (np.ndarray): 原始大图像，形状为 [H, W, C]
    #     q (deque): 轨迹队列，包含(cur_cams, p)元组
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

    if idx==0:
        # thumbnail_surface.fill((0, 0, 0))  # 填充黑色背景
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
            (60, 60, 30),
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

        # 应用逆仿射变换到裁剪矩形顶点，得到 cur_cams 和 p 的四边形坐标，并映射到缩略图
        t_mapped = apply_inverse_affine_and_map(A23t_inv, crop_corners, scale, pad_x, pad_y)  # [ (x', y'), ... ]
        # 绘制四边形
        pygame.draw.polygon(thumbnail_surface, (255, 0, 0), t_mapped, 1)   # 绘制 cur_cams 的四边形，红色

        unit_up = np.array([0, -1]) # 图像朝上的单位矢量
        q_center_mapped,q_up_end,q_rot_end = map_arrow(crop_size,A23t_inv, unit_up, scale, 2.0, pad_x, pad_y)
        if current_p is not None:
            A23p = current_p[4].cpu().numpy()  # [2, 3] 矩阵
            A23p_inv = A23inverse(A23p)
            p_mapped = apply_inverse_affine_and_map(A23p_inv, crop_corners, scale, pad_x, pad_y)  # [ (x', y'), ... ]
            pygame.draw.polygon(thumbnail_surface, (0, 255, 0), p_mapped, 1)   # 绘制 p 的四边形，绿色
            p_center_mapped,p_up_end,p_rot_end = map_arrow(crop_size, A23p_inv, unit_up, scale, 2.0, pad_x, pad_y)
            pygame.draw.polygon(thumbnail_surface, (255, 255, 0), (p_center_mapped, q_center_mapped), 1)   # 绘制 p<->q，黄色
            pygame.draw.circle(thumbnail_surface, (255, 255, 0), q_center_mapped, 2, 2)
        else:

            #draw two poly arrows x-y
            # 定义裁剪矩形的中心点和方向向量
            #draw two poly arrows x-y 定义裁剪矩形的中心点和方向向量
            # q_center_mapped,q_up_end,q_rot_end = map_arrow(crop_size,A23t_inv, unit_up, scale, 2.0, pad_x, pad_y)

            # 绘制方向轴
            draw_arrow(thumbnail_surface, (0, 255, 0), q_center_mapped, q_up_end, arrow_size=10)   # p 的朝上方向轴，绿色
            draw_arrow(thumbnail_surface, (255, 0, 0), q_center_mapped, q_rot_end, arrow_size=10)   # p 的顺时针旋转90度本体x方向轴，红色

            #draw UAV-idx
            # 创建字体对象（None 表示默认字体，字号 16）
        font = pygame.font.Font(None, 16)
        # 要绘制的文字
        text = f"GT-{idx}"
        # 渲染成 surface（白色字体）
        text_surface = font.render(text, True, (255, 255, 255))
        # 获取文字的矩形，并设置中心为 q_center_mapped
        text_rect = text_surface.get_rect(center=(int(q_center_mapped[0]), int(q_center_mapped[1])))
        # 绘制文字到 thumbnail_surface
        thumbnail_surface.blit(text_surface, text_rect)

    return thumbnail_surface

def draw_video_image(frame):
    if frame is None:
        return None
    video_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_rgb = cv2.resize(video_rgb, (512, 512))
    video_surface = pygame.surfarray.make_surface(video_rgb.swapaxes(0,1))
    return video_surface

def draw_surface(globe_objects, globe_flag, hot_map_wh, spp, scale_hot, globe_surface, rator_mode):
    if globe_objects.shape[0] > 0:
        hot_map_w, hot_map_h = hot_map_wh
        scale, pad_x, pad_y = spp
        device = globe_objects.device
        thumbnail_objects = globe_objects[:, -7:].clone()
        # xy_grid = torch.stack((torch.meshgrid(torch.linspace(-20, 20, 400), torch.linspace(-20, 20, 400), indexing='xy')), dim=-1).to(device)
        xy_grid = torch.stack((torch.meshgrid(torch.arange(0, hot_map_w), torch.arange(0, hot_map_h), indexing='xy')), dim=-1).to(device)
        thumbnail_objects[:, [0, 2]] = thumbnail_objects[:, [0, 2]] * scale
        thumbnail_objects[:, [1, 3]] = thumbnail_objects[:, [1, 3]] * scale
        thumbnail_objects[:, 0] += pad_x
        thumbnail_objects[:, 1] += pad_y
        if rator_mode == 4:
            if len(globe_surface[rator_mode]) != globe_flag.shape[0]:
                globe_surface[rator_mode] = [None] * (globe_flag.shape[0] - len(globe_surface[rator_mode])) + globe_surface[rator_mode]
            globe_surface[rator_mode] = [(surf if flag else None) for flag, surf in zip(globe_flag, globe_surface[rator_mode])]
            draw_oriented_fan(thumbnail_objects, globe_surface[rator_mode])
        
        thumbnail_objects[:, [0, 2]] = thumbnail_objects[:, [0, 2]] * scale_hot[0]
        thumbnail_objects[:, [1, 3]] = thumbnail_objects[:, [1, 3]] * scale_hot[1]
        if rator_mode == 3:
            if len(globe_surface[rator_mode]) == 0:
                globe_surface[rator_mode] = np.zeros((0, hot_map_h, hot_map_w), dtype=np.uint8)
            globe_surface[rator_mode] = np.concatenate([batched_multivariate_normal_pdf(xy_grid, thumbnail_objects[~globe_flag]).cpu().numpy(), globe_surface[rator_mode]], axis=0)
        globe_flag[:] = True

def draw_globe_surface(thumbnail_surface, globe_surface, thumbnail_size, rator_mode):
    if len(globe_surface[rator_mode]) > 0:
        if rator_mode == 3:
            surf = np.clip(np.sum(globe_surface[rator_mode], axis=0), -1, 1)
            surf_img = (np.stack([np.clip(surf, 0, 1), np.zeros_like(surf), np.clip(-surf, 0, 1)], axis=-1) * 255).astype(np.uint8)
            surf_img = cv2.resize(surf_img, thumbnail_size)
            surface = pygame.surfarray.make_surface(surf_img.swapaxes(0,1))
            thumbnail_surface.blit(surface, (0,0))
        elif rator_mode == 4:
            for surf in globe_surface[rator_mode]:
                thumbnail_surface.blit(*surf)


import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
if __name__ == '__main__':
    # config_path = 'fly/config-DOTA.json'
    # config_path = 'fly/config-TZ-ship.json'
    # config_path = 'fly/config-mstar.json'
    # config_path = 'fly/config-SSDD.json'
    # config_path = 'fly/config-RSImage.json'
    config_path = 'fly/config-DOTA-remote.json'
    config = load_config(config_path)
    # Simulation parameters
    CROP_SIZE = (512, 512)
    OBJECT_LIMIT = 256
    QUEUE_LIMIT = 4
    MAX_SCORE_LIMIT = 256

    # Initialize FlyCrop
    images_dir=config['images_dir']
    labels_dir=config['labels_dir'] if os.path.exists(config['labels_dir']) else replace_last_path(images_dir,'labels')
    image_names = [f for f in os.listdir(images_dir) if f.endswith(('.tif','.png','.jpg')) and f[0]!='#']
    conf_thres = config.get('conf_thres', 0.25)
    iou_thres = config.get('iou_thres', 0.45)
    fly_num = max(config.get('fly_num', 4), 1)
    #images = read_tif_with_tifffile(os.path.join(images_dir,image_names[0]))
    index = random.randint(0, len(image_names) - 1)
    fly = FlyCrop(120.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flys = [fly] + [FlyCrop(fly_base=fly) for _ in range(fly_num - 1)]
    mul_n = np.ceil(np.sqrt(fly_num))
    EMPTY_SIZE = int(CROP_SIZE[0] * mul_n), int(CROP_SIZE[1] * mul_n), 3
    [fly.set_size(CROP_SIZE) for fly in flys]
    map = reload(os.path.join(images_dir,image_names[index]), fly.device)

    A23_GT = load_A23_file(Path(images_dir).parent / 'crop_txt' / Path(image_names[index]).with_suffix('.txt').name)
    assert A23_GT.shape[0] >= fly_num
    try:
        labels = torch.from_numpy(load_labels(Path(labels_dir)/ Path(image_names[index]).with_suffix('.txt').name)).to(device)   # pts
        labels[:, 1:] = (labels[:, 1:].reshape(-1, 4, 2) * torch.as_tensor([[map[0].shape[1], map[0].shape[0]]], device=device)).reshape(-1, 8)
    except:
        labels = torch.zeros((0, 9), dtype=torch.float32).to(device)


    for i, fly in enumerate(flys):
        fly.link_map(map)  # Replace with your large map image path
        fly.set_affine(A23_GT[i])

    flys_q = [deque(maxlen=QUEUE_LIMIT) for _ in range(fly_num)]

    # Initialize Pygame
    pygame.init()
    window_width, window_height = 1280, 800
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Real-Time FlyCrop Simulation")

    # Font for displaying text
    font = pygame.font.SysFont(None, 24)
    [fly.init_params() for fly in flys]
    # Placeholder model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize queue
    globe_objects = torch.zeros(0, 10+6 + 1).to(device) #[n_total, 4*2+1+1 + 6(xyabcs) + 1]
    globe_scores = torch.zeros(0, 3).to(device) #[time, 4*2+1+1 + 6(xyabcs) + 1]
    menu = None
    iouv = torch.linspace(0.5, 0.95, 10).to(device)



    # 定义pygame_menu主题（全局字体大小）
    custom_theme = pygame_menu.themes.THEME_DARK.copy()
    custom_theme.widget_font = pygame.font.SysFont("Courier New", 12)
    custom_theme.widget_font_size = 12      # 组件字体大小
    custom_theme.title_font_size = 12       # 标题字体大小
    custom_theme.widget_alignment = pygame_menu.locals.ALIGN_LEFT

    #must keep same with train.py
    # model = RemoteVitModel(vit_head_out_num=1024, dudv_size=32, hidden_size_xcyc=0, hidden_size_rrotation=128,C=256).to(device)
    if config['model_best_path'] and os.path.exists(config['model_best_path']):
        model = attempt_load(config['model_best_path'], map_location=device)  # load FP32 model
        remote_model = RemoteVitModel(vit_head_out_num=1024, dudv_size=32, hidden_size_xcyc=0, hidden_size_rrotation=128,C=256).to(device)
        ckpt = torch.load(config['remote_model_path'], weights_only=True)
        remote_model.load_state_dict(ckpt['model_state_dict'])
        del ckpt
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        dict_name = {i: v for i, v in enumerate(names)}
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
        report_size=(840, 200)
        report_surface = pygame.Surface((report_size[0], report_size[1]))
        # 创建字体对象（默认字体，字号 24）
        # font = pygame.font.Font(None, 24)
        font = pygame.font.SysFont("Courier New", 14)
        mode_name = ['normal','view','detect', 'hotmap', 'sector']
        view_mult = [0, 3.5, 2.0, 0, 0]
        hot_map_shape = (200, 200)
        hot_map_w, hot_map_h  = hot_map_shape
        def reset():
            globe_surface = [[] for _ in range(len(mode_name))]
            globe_flag = torch.zeros(0).bool().to(device) 
            return globe_surface, globe_flag
        globe_surface, globe_flag = reset()
        rator_mode = 0

        thumbnail_width, thumbnail_height = thumbnail_surface.get_size()
        map_height, map_width = map[0].shape[0], map[0].shape[1]
        # 计算缩放因子，保持长宽比
        scale = min(thumbnail_width / map_width, thumbnail_height / map_height)
        # 计算偏移量以居中
        pad_x = (thumbnail_width - map_width * scale) / 2
        pad_y = (thumbnail_height - map_height * scale) / 2
        report = deque(maxlen=10)

        
        # 计算缩放因子
        scale_hot = hot_map_w / thumbnail_width, hot_map_h / thumbnail_height
        flash = True
        while running and frame_count < 99999999:
            # Limit to 60 FPS
            clock.tick(60)
            fps = clock.get_fps()
            events = pygame.event.get()
            for event in events:
                if event.type == QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    reset_flag = True
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE or event.key == pygame.K_LCTRL:
                        if event.key == pygame.K_SPACE:
                            index = (index + 1) % len(image_names)
                        else:
                            index = (index + len(image_names) - 1) % len(image_names)
                        map = reload(os.path.join(images_dir,image_names[index]), fly.device)  # Replace with your large map image path
                        A23_GT = load_A23_file(Path(images_dir).parent / 'crop_txt' / Path(image_names[index]).with_suffix('.txt').name)
                        assert A23_GT.shape[0] >= fly_num
                        for i, fly in enumerate(flys):
                            fly.link_map(map)  # Replace with your large map image path
                            fly.set_affine(A23_GT[i])
                        [q.clear() for q in flys_q]
                        [fly.init_params() for fly in flys]
                        thumbnail_width, thumbnail_height = thumbnail_surface.get_size()
                        map_height, map_width = map[0].shape[0], map[0].shape[1]
                        try:
                            labels = torch.from_numpy(load_labels(Path(labels_dir)/ Path(image_names[index]).with_suffix('.txt').name)).to(device)
                            labels[:, 1:] = (labels[:, 1:].reshape(-1, 4, 2) * torch.as_tensor([map_width, map_height], device=device)).reshape(-1, 8)   # pts
                        except:
                            labels = torch.zeros((0, 9), dtype=torch.float32).to(device)

                        # 计算缩放因子，保持长宽比
                        scale = min(thumbnail_width / map_width, thumbnail_height / map_height)
                        # 计算偏移量以居中
                        pad_x = (thumbnail_width - map_width * scale) / 2
                        pad_y = (thumbnail_height - map_height * scale) / 2
                        #
                        globe_objects = torch.zeros(0, 10+6 + 1).to(device) #[0, 17 = 4*2+1(conf)+1(cls) + 6(xyabcs) + 1]
                        globe_scores = torch.zeros(0, 3).to(device) #[0, 3]
                    elif event.key==pygame.K_RIGHT:
                        rator_mode = (rator_mode+1) % len(mode_name)
                    elif event.key==pygame.K_LEFT:
                        rator_mode = (rator_mode-1+len(mode_name)) % len(mode_name)
                    else:
                        if event.key == pygame.K_F5:  # 按下 F5 键截图
                            count_snap = 0
                            while True:
                                # snap = f'screenshot{f"_{count_snap:2d}"}.png'
                                snap = f'{Path(config_path).stem}{f"_{count_snap:2d}"}.png'
                                count_snap += 1
                                if not os.path.exists(snap):
                                    break
                            pygame.image.save(screen, snap)
                        reset_flag = False
                    if reset_flag:
                        flash = True
                        globe_surface, globe_flag = reset()

            if not flash:
                if menu is not None:
                    screen.fill((0, 0, 0))  # Clear screen
                    # 要绘制的文字
                    text = f"map:{image_names[index]}[{map_width}x{map_height}] {index}/{len(image_names)}  {rator_mode}-{mode_name[rator_mode]} fps={fps:.2f}"
                    # 渲染成 surface（yellow字体）
                    text_surface = font.render(text, True, (255, 255, 0))
                    # 获取文字的矩形，并设置中心为 p_center_mapped
                    text_rect = text_surface.get_rect(topleft=(610, 10))
                    
                    if video_surface:
                        screen.blit(video_surface, (0, 0))
                    screen.blit(thumbnail_surface, (600, 0))
                    screen.blit(text_surface, text_rect)
                    screen.blit(report_surface, (0, screen.get_height()-report_surface.get_height()))
                    menu.clear()
                    menu.add.label('1', wordwrap=False)
                    menu.update(events)
                    menu.draw(screen)  # 渲染

                pygame.display.flip()
                continue
            flash = False   # flash = True Out

            # Update FlyCrop
            image = None
            cur_cams = []
            pre_cams = []
            image_shows = np.zeros(EMPTY_SIZE, dtype=np.uint8)
            labels_mask = []
            for idx, fly in enumerate(flys):
                image = fly.render() if image is None else torch.cat([image, fly.render()], dim=0) #image[B,C,H,W]

                # Handle yaw rotation
                # if random.uniform(0, 1) < 0.03:
                #     dth = math.pi * random.uniform(-10, 10) / 180
                #     fly.rot(dth)
                
                # # Update speed
                # if random.uniform(0, 1) < 0.02:
                #     fly.update_speed(0.2)
                
                # # Motion
                # fly.motion()
                
                # Convert A23 to cur_cams
                A23t = fly.mA.squeeze(0)
                t_cos, t_sin, t_s, t_xc, t_yc = A232rot_torch(A23t, CROP_SIZE)
                cur_cams.append((t_xc, t_yc, t_cos, t_sin, A23t))
                # A23t_inv = A23inverse_torch(A23t)
                if labels.shape[0] > 0:
                    gt_pts = labels[:, 1:].clone().reshape(-1, 4, 2)
                    gt_pts = torch.cat([gt_pts, torch.ones_like(gt_pts[:, :, :1])], dim=-1)
                    gt_pts = torch.einsum('ijk, km->ijm', gt_pts, A23t.T).mean(1)   # nt, 4, 2 --> nt, 2
                    labels_mask.append(((gt_pts >= torch.as_tensor([[0, 0]], device=device)) & (gt_pts <= torch.as_tensor([CROP_SIZE], device=device))).all(-1))
            if labels.shape[0] > 0:
                labels_mask = torch.stack(labels_mask, dim=-1).amax(-1)

            # Prepare image batch for prediction (dummy)
            if image is not None:
                img = image.clone().to(device)[:, [2,1,0]].contiguous()
                img = img / 255.0  # 0 - 255 to 0.0 - 1.0
                #img[c=3,h,w]
                if len(img.shape) == 3:
                    img = img[None]  # expand for batch dim
                #img[b=cams,3,H,W]
                # Inference
                
                with torch.no_grad():  # 确保不跟踪梯度
                    objp, xcp, ycp, dup, dvp = remote_model.predict(image.clone().to(device) / 255.0)
                    pred = detect(model, img, False, conf_thres, iou_thres, mname=mname,agnostic_nms=False,classes=None,max_det=3000)                
                xcp *= map_width
                ycp *= map_height
                A23p = xcycdudv2A23_torch(xcp, ycp, dup, dvp, CROP_SIZE)
                for xcp_, ycp_, dup_, dvp_, A23p_ in zip(xcp, ycp, dup, dvp, A23p):
                    pre_cams.append([xcp_, ycp_, dup_, dvp_, A23p_])
                #
                # Process predictions                
                for idx, det in enumerate(pred):  # per image batch循环
                    image_show = image[idx].permute(1, 2, 0).contiguous().to(device='cpu').numpy()#.astype(np.uint8) #image[1,C,H,W]->[C,H,W]->image_show[H,W,C]
                    list_cls_objs = []
                    if len(det): #det[np,4(pts)*2+1(conf)+1(cls)]
                        # Rescale boxes from img_size to image_show size
                        det[:, :8] = scale_coords_poly(img.shape[2:], det[:, :8], image_show.shape).round()
                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            list_cls_objs.append((c,f" {n} {names[int(c)]}{'s' * (n > 1)}"))  # add to string
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            xyxy = [x.cpu() for x in xyxy]
                            label = '{} {:.2f}'.format(names[int(cls)], conf)
                            dir_line = 1
                            line_thickness = 2
                            color = COLORS[int(cls)%len(COLORS)]
                            if mname in [0, 1]:
                                plot_one_box(np.array(xyxy), image_show, color=color, label=label)
                            else:
                                plot_one_rot_box(np.array(xyxy), image_show, color=color, label=label, dir_line=dir_line, line_thickness=line_thickness)
                        det_new = torch.zeros((det.shape[0], 10), device=device, dtype=det.dtype) #det_new[np,4(pts)*2+1(conf)+1(cls)]
                        A23p = pre_cams[idx][-1] #(t_xc, t_yc, t_cos, t_sin, A23p)
                        A23p_inv = A23inverse_torch(A23p)
                        if mname in [0, 1]:
                            det_pts = det[:, [0, 1, 2, 1, 2, 3, 0, 3]].view(-1, 4, 2)   # x1y1 x2y1, x2y2, x1y2
                        else:
                            # 计算逆仿射矩阵
                            det_pts = det[:, :8].view(-1, 4, 2)
                        det_pts = torch.cat([det_pts, torch.ones_like(det_pts[:, :, :1])], dim=-1)
                        det_pts = torch.einsum('ijk, km->ijm', det_pts, A23p_inv.T)
                        det_new[:, :8] = det_pts.view(-1, 8)
                        det_new[:, -2:] = det[:, -2:]#[np,10=8+1+1]
                        det_side = torch.randint(0, 2, (det_new.shape[0], 1), dtype=torch.float32).to(device) #torch.zeros([det_new.shape[0],1]).to(device)
                        det_cov = pts2cov(det_new) # det_cov[np,6=(cx, cy, a, b, cos,sin)] 中心xy + abcs
                        det_cov[:,2:4] = torch.abs(det_cov[:,2:4])
                        if globe_objects.shape[0] > 0: #globe_objects[np_globe,4*2+1(conf)+1(cls) + 6(xyabcs) + 1]
                            globe_pts_conf_cls = globe_objects[:,:10] #globe_pts_conf_cls[np_globe,10]
                            globe_det_cov = globe_objects[:,10:-1] #globe_det_cov[np_globe,6(xyabcs)]
                            globe_det_side = globe_objects[:,-1:] #globe_det_side[np_globe,1(side)]
                            iou = probiou_fly_gauss(det_cov, globe_det_cov).to(det_cov.device) #iou[np,np_globe]
                            iou_mask = iou > iou_thres #det_cov[np,6(xyabcs)]-->iou_mask[np,np_globe]
                            cls_ = det_new[:, -1:] == globe_pts_conf_cls[:, -1] #cls_[np,np_globe]
                            iou_cls_mask = iou_mask.to(det_cov.device) & cls_ #c[np,np_globe]
                            #
                            mask_globe = torch.argmax(iou * iou_cls_mask, dim=1) #mask_globe[np_globe]
                            mask_det_iou = iou_cls_mask.any(-1)
                            det_side[mask_det_iou] = globe_det_side[mask_globe[mask_det_iou]] #det_side
                            #
                            conf_mask = (det_new[:, -2:-1] < globe_pts_conf_cls[:, -2]) * iou_cls_mask #conf_mask[np,np_globe]
                            max_conf = torch.argmax(conf_mask * globe_pts_conf_cls[:, -2:-1].T, dim=1)
                            mask_det = conf_mask.any(-1)
                            #
                            max_conf = max_conf[mask_det]    # 容器内置信度 大于 新标签的
                            det_new[mask_det] = globe_pts_conf_cls[max_conf]  # 提取容器内大于新标签的
                            det_cov[mask_det] = globe_det_cov[max_conf]  # 提取容器内大于新标签的
                            iou_cls_mask = iou_cls_mask.any(0)                            
                            if globe_flag.shape[0] == 0:
                                globe_flag = torch.cat([torch.zeros([globe_objects.shape[0]], dtype=torch.bool).to(device), globe_flag], dim=0)
                            globe_objects = globe_objects[~iou_cls_mask] #globe_objects[np_globe,4(pts)*2+1(conf)+1(cls) + 6(xyabcs) + 1]
                            globe_flag = globe_flag[~iou_cls_mask]
                            for surf_i in range(len(mode_name)):
                                if surf_i == rator_mode:
                                    if len(globe_surface[surf_i]) > 0:
                                        globe_surface[surf_i] = [j_ for i_, j_ in enumerate(globe_surface[surf_i]) if not iou_cls_mask[i_]]
                                        if (surf_i == 3) and (len(globe_surface[surf_i]) > 0):
                                            globe_surface[surf_i] = np.stack(globe_surface[surf_i], axis=0)
                        else:
                            mask_det_iou = torch.zeros([det_new.shape[0]],dtype=bool) #[np]
                        
                        obj17 = torch.cat([det_new,det_cov,det_side], dim=1) #obj17[np,17=10(4*2+1(conf)+1(cls))+6(xyabcs) + 1(side)]
                        if database is not None:
                            if obj17.shape[0] > 0:
                                database.insert_record(results=obj17.cpu().numpy(),
                                                        names=names,
                                                        fly_idx=idx,
                                                        image_name=image_names[index],
                                                    )
                        globe_objects = torch.cat([obj17, globe_objects], dim=0) #globe_objects[ntotal,17=10(4*2+1(conf)+1(cls))+6(xyabcs) + 1(side)]
                        globe_flag = torch.cat([torch.zeros([obj17.shape[0]], dtype=torch.bool).to(device), globe_flag], dim=0)
                        draw_surface(globe_objects, globe_flag,
                                    hot_map_wh=hot_map_shape,
                                    spp=(scale, pad_x, pad_y),
                                    scale_hot=scale_hot,
                                    globe_surface=globe_surface,
                                    rator_mode=rator_mode)
                        
                        # if len(globe_surface[rator_mode]) >= 256:
                        #     print()
                        globe_objects = globe_objects[:OBJECT_LIMIT] #globe_objects[ntotal,17=10(4*2+1(conf)+1(cls))+6(xyabcs) + 1(side)]
                        globe_flag = globe_flag[:OBJECT_LIMIT]
                        globe_surface = [surface[:OBJECT_LIMIT] for surface in globe_surface]
                        if rator_mode in [3, 4]:
                            assert len(globe_surface[rator_mode]) == globe_objects.shape[0]
                        assert mask_det_iou is not None
                        det_new_report = obj17[~mask_det_iou]
                        if det_new_report.shape[0] > 0:#det_new_report[np,17=10(4*2+1(conf)+1(cls))+6(xyabcs) + 1(side)]
                            cls = det_new_report[:, 9].to(torch.int32)#第9列是cls
                            # 统计类别和数量
                            unique_classes, counts = torch.unique(cls, return_counts=True)
                            # 打印结果
                            s2 = ''
                            for c, count in zip(unique_classes.tolist(), counts.tolist()):
                                s2+= f"{count} {names[int(c)]}{'s' * (n > 1)}, " #print(f"类别 {c} 出现 {count} 次")
                            # print(s2)
                            #det_new_report[np,17=8+1+1]
                            for i, row in enumerate(det_new_report):
                                # row 是 det_new_report[i]，形状是 [17=8+1(conf)+1(cls)]
                                conf, cls = row[8].item(), int(row[9].item())  # 转为 float 和 int
                                cx, cy, a, b, cos,sin, side = [v.item() for v in row[-7:]]  # 取出最后5个元素并转为 float
                                row_dict = {
                                    'conf': conf,
                                    'cls': cls,
                                    'cx': cx,
                                    'cy': cy,
                                    'a': a,
                                    'b': b,
                                    'cos': cos,
                                    'sin': sin,
                                    'idx':idx,
                                    'side':int(side)
                                }
                                report.append(row_dict)
                            
                    # 写文字（左上角）
                    ys = 25
                    cv2.putText(image_show, f'UAV-{idx}', (10, ys), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA) # print(s)
                    for i,(cls,cls_objs) in enumerate(list_cls_objs):
                        ys += 24
                        cv2.putText(image_show, cls_objs, (10, ys), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[int(cls)%len(COLORS)], 2, cv2.LINE_AA)
                    #
                    h_0 = int(idx // mul_n * CROP_SIZE[0])
                    h_1 = h_0 + CROP_SIZE[0]
                    w_0 = int(idx % mul_n * CROP_SIZE[1])
                    w_1 = w_0 + CROP_SIZE[1]
                    image_shows[h_0:h_1, w_0:w_1, :] = image_show
                    # Print time (inference-only)
                    # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            
            # Append to queue
            [q.append(deepcopy(cur_cams[idx])) for idx, q in enumerate(flys_q)]
            
            # Visualization
            screen.fill((0, 0, 0))  # Clear screen
            
            #left top 4 videos
            video_surface = draw_video_image(cv2.resize(image_shows, (CROP_SIZE[1], CROP_SIZE[0])))
                       
                       
            thumbnail_surface.fill((0, 0, 0))  # 填充黑色背景

            draw_globe_surface(thumbnail_surface, globe_surface, thumbnail_size, rator_mode)

            # Bottom: map with trajectories
            A23p_inv = np.asarray([[1, 0, 0], [0, 1, 0]])
            for idx, fly in enumerate(flys):
                thumbnail_surface = draw_map_with_uavs(thumbnail_surface, map[0], flys_q[idx], CROP_SIZE, cur_cams[idx], pre_cams[idx], idx)
            padxy = torch.tensor([pad_x, pad_y]).to(device)
            view_dist = view_mult[rator_mode]

            for l in globe_objects:#globe_objects[np_globe,4*2+1(conf)+1(cls) + 6(xyabcs) + 1(side)]
                xyxy = l[:8].view(4, 2).cpu().numpy() #xyxy[4,2]
                cls = int(l[9])
                side = int(l[-1])
                p_mapped = apply_inverse_affine_and_map(A23p_inv, xyxy, scale, pad_x, pad_y)  # [ (x', y'), ... ]
                color = (0, 255, 0) # COLORS[cls%len(COLORS)]
                # color = color[2], color[1], color[0]
                pygame.draw.polygon(thumbnail_surface, color, p_mapped, 2)
                if rator_mode==2:
                    # 计算中心点（四点均值）
                    cen = (scale * l[10:12] + padxy).cpu().numpy()#np.mean(p_mapped, axis=0)  #cen[2]
                    ab = scale * l[12:14] #ab[2]
                    cs = l[14:16] #ab[2]
                    # 设置颜色（红 or 蓝）
                    circle_color = (255, 0, 0) if side == 0 else (0, 0, 255)
                    # 绘制空心圆圈
                    # pygame.draw.circle(thumbnail_surface, circle_color, cen.astype(int), int(view_dist*torch.max(ab)), width=1)
                    xyabcs = list(cen) + list(ab) + list(cs)
                    draw_rotated_ellipse(thumbnail_surface, circle_color, xyabcs, filled=0, mult=2.5)

            map_text = []
            if labels.shape[0] > 0:
                labels_filter = labels[labels_mask]
                if labels_filter.shape[0] > 0:
                    for l in labels_filter:
                        xyxy = l[1:9].view(4, 2).cpu().numpy() #xyxy[4,2]
                        cls = int(l[0])
                        p_mapped = apply_inverse_affine_and_map(A23p_inv, xyxy, scale, pad_x, pad_y)  # [ (x', y'), ... ]
                        color = (255, 0, 0) # COLORS[cls%len(COLORS)]
                        pygame.draw.polygon(thumbnail_surface, color, p_mapped, 2)
                    globe_objects_xywhr = torch.cat([xyxyxyxy2xywhr(globe_objects[:, :8]), globe_objects[:, 8:10]], dim=-1)
                    correct = process_batch_obb(globe_objects_xywhr, labels_filter, iouv)
                    stats = [correct.cpu().numpy(), globe_objects[:, 8].cpu().numpy(), globe_objects[:, 9].cpu().numpy(), labels_filter[:, 0].cpu().numpy()]
                    p, r, ap, f1, ap_class, threshs, py = ap_per_class(*stats, plot=False, names=dict_name)
                    nt = np.bincount(stats[3].astype(np.int64), minlength=len(names))
                    ap50, ap595 = ap[:, 0], ap.mean(1)
                    map50, map595 = ap50.mean(), ap.mean()
                    map_text.append(('%18s' + '%11s' * 3) % ('Class', 'Labels', 'mAP@.5', 'mAP@.5:.95'))
                    pf = '%18s' + '%11i' * 1 + '%11.4g' * 2
                    map_text.append(pf % ('All', nt.sum(), map50, map595))
                    for i, idx in enumerate(ap_class):
                        map_text.append(pf % (names[idx], nt[idx], ap50[i], ap595[i]))

            if rator_mode==1:
                clss = globe_objects[:,9].to(torch.int) #clss[np]
                cen = globe_objects[:,10:12] #cen[np,2]
                assert cen.shape[0]==clss.shape[0]
                ab = globe_objects[:,12:14] #ab[np,2]
                v = globe_objects[:,14:16] #v[np,2]
                L = torch.max(ab,dim=-1) #L[np]
                arrow1 = cs2arcarrow2(v,math.radians(80.0/2)) #arrow1[np,4]
                ends1 = scale * (cen + view_dist * L[0][:,None] * arrow1[:,:2]) + padxy
                ends2 = scale * (cen + view_dist * L[0][:,None] * arrow1[:,2:]) + padxy #arrow[np,4]
                ends1 = tuple(ends1.cpu().numpy().astype(int))
                ends2 = tuple(ends2.cpu().numpy().astype(int))
                cen = tuple((scale*cen+padxy).cpu().numpy().astype(int))
                for start, end1,end2,cls in zip(cen, ends1,ends2,clss): #end[4]
                    color = COLORS[cls%len(COLORS)]
                    color = color[2], color[1], color[0]
                    pygame.draw.line(thumbnail_surface, color, start, end1, width=1)
                    pygame.draw.line(thumbnail_surface, color, start, end2, width=1)

            # 要绘制的文字
            text = f"map:{image_names[index]}[{map_width}x{map_height}] {index}/{len(image_names)}  {rator_mode}-{mode_name[rator_mode]} fps={fps:.2f}"
            # 渲染成 surface（yellow字体）
            text_surface = font.render(text, True, (255, 255, 0))
            # 获取文字的矩形，并设置中心为 p_center_mapped
            text_rect = text_surface.get_rect(topleft=(610, 10))
            # 绘制文字到 thumbnail_surface
            # thumbnail_surface.blit(text_surface, text_rect)
            # screen.blit(thumbnail_surface, (600, 0))

            #side number
            globe_side = globe_objects[:,-1] #globe_side[np]
            num_side = globe_side.sum().item()  # red 0, blue 1
            num_side = (globe_side.shape[0] - num_side,num_side)
            score3 = ((globe_objects[:, -5] * globe_objects[:, -4]) * (1 + -2 * globe_side)).sum().item() if globe_side.shape[0] > 0 else 0
            #
            score = torch.tensor([num_side[0],num_side[1], score3]).long().to(device)[None] #score[1,3]
            globe_scores = torch.cat([globe_scores,score], dim=0)[-MAX_SCORE_LIMIT:] #globe_scores[time,3]

            #left buttom draw
            # big_crop_size = (1024,1024)
            # A23Tct,A23padt = A23Crop_torch(A23t,CROP_SIZE,big_crop_size)
            # # 3. 渲染到 report_surface 上
            # report_surface.blit(pad_croped_image_rgb, (0, 0))
            # draw_surface(report_surface, None, A23padt, CROP_SIZE, report_size[0] / big_crop_size[0], colorp=(255, 128, 0), colort=(255, 255, 0)) #yellow: gt
            # 要显示的文字
            # report文字字符串（多行）
            # 组装要显示的文字字符串（多行）
            line_surfaces = []
            header = f"{'time':<20}{'sd':<3}{'cls':<4}{'name':<20}{'conf':<4} {'cx':<7} {'cy':<7} {'a':<6} {'b':<6} {'angle':<8} {'uav':<3} {'map':<12}"
            surface = font.render(header, True, (255,255,255))  # 白色字体
            line_surfaces.append((surface, (5, 10)))  # 位置偏移
            # 获取当前时间
            now = datetime.now()
            # 转换为字符串，格式为 年-月-日 时:分:秒
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")

            map_name = os.path.splitext(image_names[index])[0]
            text_lines = []
            for i, item in enumerate(report):
                cls = item['cls']
                idx = item['idx']
                cos_t,sin_t = item['cos'],item['sin']
                angle = np.arctan2(sin_t,cos_t)*180/np.pi
                side = int(item['side'])
                line = f"{time_str:<20}{side:<3}{cls:<4}{names[cls]:<20}{item['conf']:<4.2f} {item['cx']:<7.1f} {item['cy']:<7.1f} {item['a']:<6.2f} {item['b']:<6.2f} {angle:<8.2f} {idx:<3} {f'{index}-{map_name}':<12}"
                text_lines.append((cls,line))
            # 合并为一个大字符串，以换行分隔
            # text_str = "\n".join(text_lines)

            # 创建一个和目标大小相同的 surface，背景透明或可选背景色
            report_surface.fill((0, 0, 0))  # 填充黑色背景
            # 使用 font 对象渲染每一行（pygame 不支持自动换行，所以逐行渲染）
            line_height = font.get_linesize()  # 行高
            for i, (cls,line) in enumerate(text_lines):
                surface = font.render(line, True, swap_rb(COLORS[int(cls)%len(COLORS)]))  # 白色字体
                line_surfaces.append((surface, (5, 10 + (1+i) * line_height)))  # 位置偏移

            # 把每一行文字绘制上去
            for surface, pos in line_surfaces:
                report_surface.blit(surface, pos)
            # 把文字贴到 report_surface 上（比如左上角）
            if video_surface:
                screen.blit(video_surface, (0, 0))
            screen.blit(thumbnail_surface, (600, 0))
            screen.blit(text_surface, text_rect)
            screen.blit(report_surface, (0, screen.get_height()-report_surface.get_height()))



            if len(map_text) > 0:
                score_width, score_height = 400, 180
                menu = pygame_menu.Menu("mAP", score_width, score_height, 
                                        theme=custom_theme,  # 可选主题
                                        enabled=True,  # 允许外部控制
                                        mouse_enabled=True,  # 启用鼠标交互
                                        position=(report_surface.get_width() + 10, thumbnail_surface.get_height() + 5, False)
                )
                for mtxt in map_text:
                    menu.add.label(mtxt, wordwrap=False)
                menu.draw(screen)

            # Update the display
            pygame.display.flip()
            
            frame_count += 1

    pygame.quit()
    sys.exit()
