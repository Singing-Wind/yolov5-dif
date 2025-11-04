import cv2
import random
import os
import numpy as np
import json


def plot_one_box(x, img, color=None, label=None, line_thickness=None,text=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    if text is not None:
        cv2.putText(img, text, (c1[0], c1[1] - 2 + 20), 0, tl/3, color, thickness=tf, lineType=cv2.LINE_AA)

def plot_one_rot_box(x, img, color=None, label=None, line_thickness=None, leftop=False, radius=3, dir_line=False,text=None):
    tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    x = np.int32(x)
    leftop_x = (x[0], x[1])
    if dir_line:
        x1, y1 = (x[0] + x[2]) / 2, (x[1]+x[3]) / 2
        x2, y2 = (x[4] + x[6]) / 2, (x[5] + x[7]) / 2
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cv2.arrowedLine(img, (int(cx), int(cy)), (int(x1), int(y1)), (0, 0, 255), thickness=tl)
    x = x.reshape((-1, 1, 2))
    cv2.polylines(img, [x], True, color, thickness=tl)
    if leftop:
        cv2.circle(img, leftop_x, radius, color, tl)
    if label:
        tf = max(tl - 1, 1) 
        cv2.putText(img, label, leftop_x, 0, tl/3, color, thickness=tf, lineType=cv2.LINE_AA)
    if text is not None:
        cv2.putText(img, text, (leftop_x[0],leftop_x[1]+20), 0, tl/3, color, thickness=tf, lineType=cv2.LINE_AA)

def draw_rotated_box(pts, img, color, thickness=2):
    #在图像上绘制变换后的4点多边形
    pts = pts.astype(int).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

def draw_polygon_multicolor(img, pts, colors, thickness=2, isClosed=True):
    # 在 img 上用不同颜色绘制多边形的每一条边。
    # 参数
    # img : np.ndarray (H, W, 3)      目标图像，uint8/BGR
    # pts : np.ndarray (N, 2) or list  多边形顶点，按顺时针或逆时针排列
    # colors : list[tuple]             每条边的颜色，长度应为 N (闭合) 或 N-1 (不闭合)
    #                                   BGR 格式，例如 (0,255,0)
    # thickness : int                  线宽
    # isClosed : bool                     True=自动补最后一条边，False=不补
    pts = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
    n = len(pts)

    # 如果需要闭合多边形，把最后一条边也画出来
    edges = [(i, (i + 1) % n) for i in range(n)] if isClosed else [(i, i + 1) for i in range(n - 1)]

    if len(colors) != len(edges):
        raise ValueError(f"colors 的数量必须等于边的数量 ({len(edges)})")

    for (i, j), color in zip(edges, colors):
        cv2.line(img,
                 tuple(pts[i]),
                 tuple(pts[j]),
                 color=color,
                 thickness=thickness,
                 lineType=cv2.LINE_AA)

def draw_frame(im,HW, A23_r,thickness=2):
    H, W = im.shape[:2] if HW==None else HW
    # 图像四个顶点：左上、右上、右下、左下
    xyxy_img = np.array([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H]
    ], dtype=np.float32)  # shape: (4, 2)
    # 构造齐次坐标 [x, y, 1]
    ones = np.ones((4, 1), dtype=np.float32)
    xy_homo = np.hstack([xyxy_img, ones])  # shape: (4, 3)
    # 仿射逆变换
    xyxyp = (A23_r @ xy_homo.T).T  # shape: (4, 2)
    # 转为 int32 以便绘图
    pts = xyxyp.astype(np.int32).reshape(-1, 1, 2)
    draw_polygon_multicolor(im, [pts], colors=[(0,0,255),(255,255,255),(255,255,255),(0,255,0)], thickness=thickness)
    #img, pts, colors, thickness=2
def draw_polygon_image_objs(im, HW, labels,colors,A23_r,thickness=2):
    for (cls_, *xyxy) in labels:
        pts = np.array(xyxy, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(im, [pts], isClosed=True, color=colors[int(cls_) % len(colors)], thickness=thickness)
    #
    draw_frame(im,HW,A23_r,thickness=thickness)



def plot_images_from_8points():
    image_path = '/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA768/val/images'
    label_path = '/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA768/val/labelTxt1.5'
    images = os.listdir(image_path)
    save_path = './runs/detect/results'

    for image in images:
        # 读取label
        label_name = image.split('.')[0] + '.txt'
        src_img = cv2.imread(os.path.join(image_path, image))
        with open(os.path.join(label_path, label_name), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(' ')
                segmentation = [int(float(x)) for x in line[:8]]
                category = line[8]
                # xmin, ymin, xmax, ymax = min(segmentation[::2]), min(segmentation[1::2]), \
                #                          max(segmentation[::2]), max(segmentation[1::2])
                # xyxy = np.array([xmin, ymin, xmax, ymax])
                # plot_one_box(xyxy, src_img, label=category)
                plot_one_rot_box(segmentation, src_img, label=category, dir_line=True, leftop=True)
            cv2.imwrite(os.path.join(save_path, image), src_img)


def plot_images_from_xywh():
    image_path = r'/home/LIESMARS/2019286190105/datasets/final-master/UCAS50/images/train'
    label_path = r'/home/LIESMARS/2019286190105/datasets/final-master/UCAS50/labels/train'
    images = os.listdir(image_path)
    save_path = '../runs/detect/exp2'
    # clses = ['triangle_horizontal', 'triangle_vertical', 'triangle_oblique', 'dangerous_goods', 'dangerous']
    for image in images:
        label_name = image.split('.')[0] + '.txt'
        src_img = cv2.imread(os.path.join(image_path, image))
        # width, height = src_img.shape[:2]
        height, width = src_img.shape[:2]
        label = os.path.join(label_path, label_name)
        if not os.path.exists(label):
            continue
        with open(label, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(' ')
                # cls = int(line[0])
                # category = clses[cls]
                category = line[0]
                xywh = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
                xyxy = np.array(xywh2xyxy(xywh, width, height))
                plot_one_box(xyxy, src_img, label=category)
            cv2.imwrite(os.path.join(save_path, image), src_img)


def plot_images_from_rot():
    image_path = r'/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA1.0-1.5/val/images'
    label_path = r'/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA1.0-1.5/val/labels1.5'
    save_path = r'/home/LIESMARS/2019286190105/finalwork/yolov5/runs/detect/images'
    images = os.listdir(image_path)
    for image in images:
        print(image)
        label_name = image.split('.')[0] + '.pts'
        src_img = cv2.imread(os.path.join(image_path, image))
        height, width  = src_img.shape[:2]
        with open(os.path.join(label_path, label_name), 'r', encoding='utf-8') as f:
            for line in f:
                points = line.strip().split(' ')
                arr = []
                for i, x in enumerate(points[1:]):
                    if i % 2 == 0:
                        arr.append(float(x) * width)
                    else:
                        arr.append(float(x) * height)
                plot_one_rot_box(np.array(arr), src_img, dir_line=True, label=points[0], leftop=True)
                # plot_one_rot_box(np.array(arr), src_img, dir_line=False, label=None)
            cv2.imwrite(os.path.join(save_path, image), src_img)


def xyrot2xy(x, width, height):
    x = np.array(x, dtype=np.float)
    x[:, 0] *= width
    x[:, 1] *= height
    return x


def xywh2xyxy(x, width, height):
    x, y, w, h = x[0], x[1], x[2], x[3]
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    x1 *= width
    x2 *= width
    y1 *= height
    y2 *= height
    return [int(x1), int(y1), int(x2), int(y2)]


if __name__ == '__main__':
    # plot_images_from_xywh()
    plot_images_from_8points()
    # plot_images_from_rot()
