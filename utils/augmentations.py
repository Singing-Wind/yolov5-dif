# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Image augmentation functions
"""

import logging
import math
import random

import cv2
import numpy as np

from utils.general import colorstr, segment2box, resample_segments, check_version, xyxyxyxy2xywhr, xywhr2xyxyxyxy
from utils.metrics import bbox_ioa, bbox_ioas, batch_probiou
from DOTA_devkit.polyiou_cpu import poly_iou_cpu64

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        hue, sat, val = cv2.split(cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2HSV))
        hue0, sat0, val0 = cv2.split(cv2.cvtColor(im[..., 3:], cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))

        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        lut_hue0 = ((x * r[0]) % 180).astype(dtype)
        lut_sat0 = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val0 = np.clip(x * r[2], 0, 255).astype(dtype)
        im_hsv0 = cv2.merge((cv2.LUT(hue0, lut_hue0), cv2.LUT(sat0, lut_sat0), cv2.LUT(val0, lut_val0)))
        # cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed
        # im[:] = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        im_ = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)  # no return needed
        im0 = cv2.cvtColor(im_hsv0, cv2.COLOR_HSV2BGR)  # no return needed
        im[:] = np.concatenate([im_, im0], axis=-1)


def hist_equalize(im, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    # Replicate labels
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def random_perspective(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, flip = [0.5,0.5],
                       border=(0, 0), iou_thr = 0.3, clip_rate=0.2, bkgrd=None):
    mosaic_mode = border[0]<0 and border[1]<0
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale) #1.0/(1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Flip (mirror) transformation
    Fud = np.eye(3)
    if random.uniform(0,1) < flip[1]:#flipud
        Fud[1, 1] = -1  # flip horizontally
        Fud[1, 2] = height  # adjust translation to keep image within bounds
    Flr = np.eye(3)
    if random.uniform(0,1) < flip[0]:#fliplr
        Flr[0, 0] = -1  # flip horizontally
        Flr[0, 2] = width  # adjust translation to keep image within bounds

    # Combined rotation matrix
    M = Fud @ Flr @ T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if bkgrd is not None:
            assert bkgrd.shape==im.shape or (2*bkgrd.shape[0]==im.shape[0] and 2*bkgrd.shape[1]==im.shape[1])
            im_dst = bkgrd.copy()
            methods = [cv2.INTER_NEAREST,cv2.INTER_LINEAR,cv2.INTER_AREA]
            cv2.warpAffine(im, M[:2], dsize=(width, height), dst=im_dst, borderMode=cv2.BORDER_TRANSPARENT, flags=random.choice(methods))
            im = im_dst
        else:
            if perspective:
                im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                methods = [cv2.INTER_NEAREST,cv2.INTER_LINEAR,cv2.INTER_AREA]
                im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114), flags=random.choice(methods),borderMode=cv2.BORDER_REFLECT_101)

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        hbox = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                hbox[i] = segment2box(xy, width, height)
            ioa = np.ones((n),dtype=float)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            
            if targets.shape[-1] == 9:
                xy[:, :2] = targets[:, 1:].reshape(n * 4, 2)  # xy4
            else:
                xy[:, :2] = targets[:, [1, 2, 1, 4, 3, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                # iou_thr = 0.2
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            i,ioa = out_range_filt_new(xy, (im.shape[1], im.shape[0]), iou_thr)
            xy,ioa = xy[i],ioa[i]
            xy = clip_xywhr_rboxes(xy, 0, 0, width, height, clip_rate) #xy[nt,8(xyxyxyxy)]
            #only for hbb
            if targets.shape[-1]==5: #1+4(xywh)
                x = xy[:, [0, 2, 4, 6]] #x[nt,4]
                y = xy[:, [1, 3, 5, 7]] #y[nt,4]
                hbox = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, -1).T #hbox[nt,4(xyxy)]
                # clip: hbox[nt,4(xyxy)]->hbox[nt,4(xyxy)]
                hbox[:, [0, 2]] = hbox[:, [0, 2]].clip(0, width)
                hbox[:, [1, 3]] = hbox[:, [1, 3]].clip(0, height)
            else:#obb
                assert targets.shape[-1] == 9 #1+4(pts)*2
            targets = targets[i]

        # filter candidates
        #i = box_candidates_old(box1=targets[:, 1:5].T * s, box2=hbox.T, area_thr=0.01 if use_segments else 0.10)
        #i = box_candidates(boxes=hbox, width=width, height=height)        
        if targets.shape[-1] == 5: #hbb
            if mosaic_mode==0:
                ind = box_candidates_ioa(boxes=hbox, width=width, height=height, iou_thr=iou_thr)
                out_xyxy = hbox[ind]
                targets = targets[ind]
                ioa = ioa[ind]
                assert ioa.shape[0]==targets.shape[0]
            else:
                out_xyxy = hbox
        else: #obb
            out_xyxy = xy #xy[nt,8]->out_xyxy[nt,8]
            
        targets[:, 1:] = out_xyxy #targets[nt,9=1+8]
    else:
        ioa = np.empty((0),dtype=float)
    
    assert ioa.shape[0]==targets.shape[0]
    return im, targets,ioa, M[:2] #im[H,W,C] targets[nt,9=1+8]


def copy_paste(im, labels, segments, p=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int64)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def cutout(im, labels, p=0.5):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates_old(box1, box2, width, height, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    #
    cx,cy = (box1[2] + box1[0])/2, (box1[3] + box1[1])/2
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (cx>=0) & (cx<=width) & (cy>=0) & (cy<=height) & (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

def box_candidates(boxes, width, height):  # box1(n,4), box2(n,4)
    cx,cy = (boxes[:,2] + boxes[:,0])/2, (boxes[:,3] + boxes[:,1])/2 #cx[n],cy[n]
    return (cx>=0) & (cx<=width) & (cy>=0) & (cy<=height)# candidates

#bbox_ioas
def box_candidates_ioa(boxes, width, height, iou_thr=0.2, w_min=0, h_min=0):  # boxes(n,4)
    image_box = np.array([w_min,h_min, width,height], dtype=np.float64)#image_poly[8]
    iou = bbox_ioas(image_box,boxes) #boxes[nt,4] -> iou[nt]
    assert np.all((iou >= 0) & (iou <= 1.0001)), "iou 中存在不在 [0, 1] 范围内的值"
    return iou>iou_thr #(cx>=0) & (cx<=width) & (cy>=0) & (cy<=height)# candidates

def out_range_filt_new(poly_ptses, shape, iou_thresh=0.2,ioa4=None): #poly_ptses[nt,8=4*2(xyxyxyxy)]
    assert ioa4 is None or ioa4.shape[0]==poly_ptses.shape[0]
    if len(shape) == 2:
        shape = [0, 0, shape[0], shape[1]]  # x1y1x2y2
    image_poly = np.array([shape[0], shape[1], shape[2], shape[1], shape[2],shape[3], shape[0],shape[3]], dtype=np.float32).reshape(1, -1)#image_poly[8]
    image_area = (shape[2] - shape[0]) * (shape[3] - shape[1])
    # v11
    # image_poly = xyxyxyxy2xywhr(image_poly.astype(np.float32))
    # poly_ptses = xyxyxyxy2xywhr(poly_ptses.astype(np.float32))
    # iou = batch_probiou(poly_ptses, image_poly).cpu().numpy().reshape(-1)
    # obj_area = poly_ptses[:, 2] * poly_ptses[:, 3]
    # v5
    iou = poly_iou_cpu64(poly_ptses.astype(np.float64), image_poly.astype(np.float64)).reshape(-1) #iou[nt] for image shape
    obj_area = cal_poly_area_array(poly_ptses) # obj_area[nt] 要排好顺序才行

    sec_area = iou * (image_area + obj_area) / (1 + iou) #sec_area[nt] for image shape
    tmp = obj_area <= 0
    sec_area[tmp] = 0
    obj_area[tmp] = 1
    ioa = sec_area / obj_area #ioa[nt]
    if ioa4 is not None:
        assert ioa4.shape[0]==ioa.shape[0]
        ioa = np.minimum(ioa,ioa4)
    polys_out_ids = np.where(ioa > iou_thresh)[0] #polys_out_ids[nt]
    return polys_out_ids,ioa #polys_out_ids[nt] ioa[nt]

def cal_poly_area_array(ptses):
    points = ptses.reshape(-1,4,2)
    # 在数组末尾添加第一个点
    points = np.concatenate([points, points[:, 0:1, :]], axis=-2)
    # 计算顺时针和逆时针相邻点乘积之和
    clockwise_sum = np.sum(points[:, :-1, 0] * points[:, 1:, 1], axis=-1)
    counterclockwise_sum = np.sum(points[:, :-1, 1] * points[:, 1:, 0], axis=-1)
    # 计算围成的面积
    area = abs(clockwise_sum - counterclockwise_sum) / 2
    return area


def resample_segments_v11(segments, n=100):
    """
    Inputs a list of segments (n,2) and returns a list of segments (n,2) up-sampled to n points each.

    Args:
        segments (list): a list of (n,2) arrays, where n is the number of points in the segment.
        n (int): number of points to resample the segment to. Defaults to 1000

    Returns:
        segments (list): the resampled segments.
    """
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # segment xy
    return segments

def clip_xywhr_rboxes(xy, x1, y1, x2, y2, clip_rate=0.0):
    margin = int(clip_rate * min(x2-x1,y2-y1))
    assert margin>=0
    assert x1<x2 and y1<y2
    segments = [x for x in xy.copy().reshape(-1, 4, 2)]
    segments = resample_segments_v11(segments)
    bboxes = []
    for segment in segments:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        segment = segment.reshape(-1, 2)
        # if np.array([x.min() < 0, y.min() < 0, x.max() > x2, y.max() > y2]).sum() >= 3:
        x, y = segment.T  # segment xy
        inside = (x >= x1-margin) & (y >= y1-margin) & (x <= x2+margin) & (y <= y2+margin)
        if inside.sum() > 0:
            points_inside = segment[inside]
            x1a, x2a, y1a, y2a = points_inside[:, 0].min(), points_inside[:, 0].max(), points_inside[:, 1].min(), points_inside[:, 1].max()
            
            segment[:, 0] = segment[:, 0].clip(x1a, x2a)
            segment[:, 1] = segment[:, 1].clip(y1a, y2a)

            (cx, cy), (w, h), angle = cv2.minAreaRect(segment)
            bboxes.append([cx, cy, w, h, angle / 180 * np.pi] if (w * h) > 0 else [0, 0, 0, 0, 0])
        else:
            bboxes.append([0, 0, 0, 0, 0])
    xy = xywhr2xyxyxyxy(np.array(bboxes, dtype=np.float32)).reshape(-1, 8)
    return xy