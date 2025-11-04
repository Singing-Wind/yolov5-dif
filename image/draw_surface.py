
import numpy as np
import pygame

def draw_arrow(surface, color, start, end, arrow_size=10):
    # 在 Pygame surface 上绘制一个箭头。
    # 参数:
    #     surface (pygame.Surface): 要绘制箭头的表面。
    #     color (tuple): 箭头的颜色，格式为 (R, G, B)。
    #     start (tuple): 箭头起点坐标 (x, y)。
    #     end (tuple): 箭头终点坐标 (x, y)。
    #     arrow_size (int, 可选): 箭头的大小。默认值为 10。
    pygame.draw.line(surface, color, start, end, 2)  # 主线

    # 计算箭头方向
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    if length == 0:
        return
    direction = direction / length

    # 计算两个箭头边的点
    angle = np.pi / 6  # 30 度
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    arrow_point1 = end - arrow_size * direction + arrow_size * np.dot(rot_matrix, direction)
    arrow_point2 = end - arrow_size * direction + arrow_size * np.dot(rot_matrix, -direction)
    # pygame.draw.line(surface, color, end, tuple(arrow_point1), 2)
    # pygame.draw.line(surface, color, end, tuple(arrow_point2), 2)