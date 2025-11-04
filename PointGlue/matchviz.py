# matchviz.py

import cv2
import numpy as np

def plot_images_and_matches(img0, img1, kpts0, kpts1, color=(0, 1, 0)):
    # 绘制两张图像并连线匹配点
    if img0.ndim == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    out_img = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    out_img[:h0, :w0] = img0
    out_img[:h1, w0:w0 + w1] = img1

    if kpts0 is not None and kpts1 is not None:
        for p0, p1 in zip(kpts0, kpts1):
            x0, y0 = p0
            x1, y1 = p1
            pt1 = (int(x0), int(y0))
            pt2 = (int(x1 + w0), int(y1))
            cv2.line(out_img, pt1, pt2, color=(0, 255, 0), thickness=1)
            cv2.circle(out_img, pt1, 2, color=(255, 0, 0), thickness=-1)
            cv2.circle(out_img, pt2, 2, color=(0, 0, 255), thickness=-1)
    return out_img
