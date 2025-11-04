# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from DOTA_devkit.polyiou_cpu import poly_iou_cpu64


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), cut=False):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    f1 = np.zeros((nc, 1000))
    ic = np.zeros(nc)
    theshes = torch.ones(len(names)) * 0.25
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            py.append(np.ones(1000)) 
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            t_thresh = np.interp(-px, -conf[i], conf[i]) # t_thresh interp

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], cut=cut)
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

            f1[ci] = 2 * p[ci] * r[ci] / (p[ci] + r[ci] + 1e-16)#f1[nc,1000]
            ic[ci] = f1[ci].argmax()#ic[cls]
            #theshes[int(c)] = ic[ci] / px.shape[0]
            theshes[int(c)] = np.clip(t_thresh[int(ic[ci])],np.min(conf[i]),np.max(conf[i]))#ic[ci] / px.shape[0]

    # Compute F1 (harmonic mean of precision and recall)
    # f1 = 2 * p * r / (p + r + 1e-16)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    ic = np.round(ic).astype(int)
    #ap[nc, 10=tp.shape[1]]
    return p[np.arange(nc), ic], r[np.arange(nc), ic], ap, f1[np.arange(nc), ic], unique_classes.astype('int32'), theshes, py


def compute_ap(recall, precision, cut=False):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    if cut:
        mrec = np.concatenate(([0.0], recall, [recall[-1]],[1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    else:
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def process_batch_base4ConfusionMatrix(func):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        None, updates confusion matrix accordingly
    """
    def warpper(self, detections, labels):
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        ############
        iou = func(labels, detections)#iou[ngt,npred] 得到真值框和预测框的iou匹配矩阵
        #############
        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN
    return warpper

class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45, pts=False):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres
        if pts:
            self.process_batch = self._process_batch_obb
        else:
            self.process_batch = self._process_batch


    @process_batch_base4ConfusionMatrix
    def _process_batch(labels, detections):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """
        return box_iou(labels[:, 1:], detections[:, :4])#iou[ngt,npred] 得到真值框和预测框的iou匹配矩阵


    @process_batch_base4ConfusionMatrix
    def _process_batch_obb(labels, detections):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, a, class
            labels (Array[M, 9]), class, x1, y1, x2, y2, x3, y3, x4, y4
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """
        return batch_probiou(xyxyxyxy2xywhr(labels[:, 1:]), detections[:, :5])#iou[ngt,npred] 得到真值框和预测框的iou匹配矩阵

    def matrix(self):
        return self.matrix

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculates IoU, GIoU, DIoU, or CIoU between two boxes, supporting xywh/xyxy formats.

    Input shapes are box1(1,4) to box2(n,4).
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)#box1[N, 4]->area1[N]
    area2 = box_area(box2.T)#box2[M, 4]->area2[M]

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)#inter[N,M]
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area

def bbox_ioas(region, boxes, eps=1E-7):
    """ Returns the intersection over boxes area given region, boxes. Boxes are x1y1x2y2
    region:       np.array of shape(4)
    boxes:       np.array of shape(n,4)
    returns:    np.array of shape(n)
    """
    boxesT = boxes.transpose()

    # Get the coordinates of bounding boxes
    x1 , y1 , x2 , y2  = region[0], region[1], region[2], region[3]
    x1s, y1s, x2s, y2s = boxesT[0], boxesT[1], boxesT[2], boxesT[3]

    # Intersection area
    inter_area = (np.minimum(x2, x2s) - np.maximum(x1, x1s)).clip(0) * \
                 (np.minimum(y2, y2s) - np.maximum(y1, y1s)).clip(0)
    #inter_area[n]

    # boxes area
    box2_area = (x2s - x1s) * (y2s - y1s) + eps #box2_area[n]

    # Intersection over boxes area
    return inter_area / box2_area  #-->iou[n]


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=(),plot_f1=1,grid=1):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    #
    if grid:
        ax.grid(True)
    if plot_f1:
        # 定义P和R的取值范围
        P = np.linspace(0, 1, 400)
        R = np.linspace(0, 1, 400)
        P, R = np.meshgrid(P, R)
        # 计算F1值
        F1 = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P + R)!=0)  # 处理除以零
        # 绘制指定等高线的等高线图
        levels = np.linspace(0.1, 0.9, 9)
        contour = plt.contour(P, R, F1, levels=levels, colors='green', linestyles='-', linewidths=0.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    #
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()

def plot_pr_curves(pys, aps, save_path, names=(), methods=[], plot_f1=1, grid=1):
    assert len(pys) == len(aps)
    colors = ['red', 'green', 'blue', 'yellow', 'grey', 'black']
    px = np.linspace(0, 1, 1000)
    c = len(names)
    n = len(pys)
    assert methods==[] or len(methods)==n
    
    if not isinstance(save_path, Path):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # 创建一个大图
    fig, axs = plt.subplots(1, c + 1, figsize=(5 * (c + 1), 5), tight_layout=True)
    
    for j in range(n):#methods j
        py = np.stack(pys[j], axis=1)
        ap = aps[j]

        # 绘制每个类的子图
        for i in range(c):#classes i
            if grid:
                axs[i].grid(True)
            if plot_f1:
                P = np.linspace(0, 1, 400)
                R = np.linspace(0, 1, 400)
                P, R = np.meshgrid(P, R)
                F1 = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P + R) != 0)
                levels = np.linspace(0.1, 0.9, 9)
                contour = axs[i].contour(P, R, F1, levels=levels, colors='green', linestyles='-', linewidths=0.5)
                axs[i].clabel(contour, inline=True, fontsize=8, fmt='%.1f')
            axs[i].plot(px, py[:, i], linewidth=1, label=f'{methods[j]} {100*ap[i, 0]:.2f}%', color=colors[j % len(colors)])
            axs[i].set_title(names[i])
            axs[i].set_xlabel('Recall')
            axs[i].set_ylabel('Precision')
            axs[i].set_xlim(0, 1)
            axs[i].set_ylim(0, 1)
            axs[i].legend()

        # 绘制所有类的平均 PR 曲线的子图
        if grid:
            axs[c].grid(True)
        if plot_f1:
            P = np.linspace(0, 1, 400)
            R = np.linspace(0, 1, 400)
            P, R = np.meshgrid(P, R)
            F1 = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P + R) != 0)
            levels = np.linspace(0.1, 0.9, 9)
            contour = axs[c].contour(P, R, F1, levels=levels, colors='green', linestyles='-', linewidths=0.5)
            axs[c].clabel(contour, inline=True, fontsize=8, fmt='%.1f')
        axs[c].plot(px, py.mean(1), linewidth=1, label=f'{methods[j]} all classes {100 * ap[:, 0].mean():.2f}%', color=colors[j % len(colors)])
    
    axs[c].set_title('All Classes')
    axs[c].set_xlabel('Recall')
    axs[c].set_ylabel('Precision')
    axs[c].set_xlim(0, 1)
    axs[c].set_ylim(0, 1)
    axs[c].legend()
    
    # 保存图像
    fig.savefig(save_path / 'prs_comparison.png', dpi=250)
    plt.close()

def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def _get_covariance_matrix(boxes, ft=False):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    if ft:
        cos = (1 / (1 + c.pow(2))).sqrt()
        sin = c * cos
    else:
        cos = c.cos()
        sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    Calculate probabilistic IoU between oriented bounding boxes.

    Implements the algorithm from https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
        CIoU (bool, optional): If True, calculate CIoU. Defaults to False.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): OBB similarities, shape (N,).

    Note:
        OBB format: [center_x, center_y, width, height, rotation_angle].
        If CIoU is True, returns CIoU instead of IoU.
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

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
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou

def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

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
    return 1 - hd

def process_batch_base(func):
    # Return correct predictions matrix.
    # Arguments:
    #     detections
    #     labels[nt,1+4*2]
    # Returns:
    #     correct (Array[np, 7=5(xywh)+1(conf)+1(cls)]), for 10 IoU levels
    def warpper(detections, labels, iouv):
        correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)#correct[预测框数量,10级]
        ############
        iou = func(labels, detections)#iou[ngt,npred] 得到真值框和预测框的iou匹配矩阵
        #############
        x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, -1]))#x[gtid[],predid[]] IoU 足够大，类labels[:, 0:1]要能匹配
        assert x[0].shape[0]==x[1].shape[0]#==nmatched
        if x[0].shape[0]:#x[0]是gt框下标数组，x[1]是预测框下标数组，这里要求gt框数量>0
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()#matches[nmatched,3=(1(gtid)+1(predid)+1(iou))]
            if 1:
            # v5 version
                if x[0].shape[0] > 1:#gt框数量>1才搞进一步优化
                    matches = matches[matches[:, 2].argsort()[::-1]]#matches[nmatched,3=(1(gtid)+1(predid)+1(iou))]，第2列(0开始实际上是第三列)，iou按顺序从大到小重新对matches排序
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]#第1列取唯一，即预测框predid应该唯一，相同predid只保留上面iou最大的
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]#第0列取唯一，即真值框gtid应该唯一，相同gtid只保留上面iou最大的
                matches = torch.Tensor(matches).to(iouv.device)#唯一最大iou匹配三元组matches[nmatched,3=(1(gtid)+1(predid)+1(iou))]
                assert matches.shape[0]<=min(detections.shape[0],labels.shape[0])
                correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
            else:
            # v11 version
                for i, threshold in enumerate(iouv.cpu().tolist()):
                    matches_v11 = matches[matches[:, 2] >= threshold]
                    if matches_v11.shape[0]:
                        if matches_v11.shape[0] > 1:
                            matches_v11 = matches_v11[matches_v11[:, 2].argsort()[::-1]]
                            matches_v11 = matches_v11[np.unique(matches_v11[:, 1], return_index=True)[1]]
                            # matches_v11 = matches_v11[matches_v11[:, 2].argsort()[::-1]]
                            matches_v11 = matches_v11[np.unique(matches_v11[:, 0], return_index=True)[1]]
                        correct[matches_v11[:, 1].astype(int), i] = True
        return correct#correct[predid,10级] bool数组
    return warpper

@process_batch_base
def process_batch(labels, detections):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    return box_iou(labels[:, 1:], detections[:, :4])#iou[ngt,npred] 得到真值框和预测框的iou匹配矩阵


@process_batch_base
def process_batch_obb(labels, detections):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, a, class
        labels (Array[M, 9]), class, x1, y1, x2, y2, x3, y3, x4, y4
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    return batch_probiou(xyxyxyxy2xywhr(labels[:, 1:]), detections[:, :5])#iou[ngt,npred] 得到真值框和预测框的iou匹配矩阵


def xyxyxyxy2xywhr(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation]. Rotation values are
    returned in radians from 0 to pi/2.

    Args:
        x (numpy.ndarray | torch.Tensor): Input box corners [xy1, xy2, xy3, xy4] of shape (n, 8).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted data in [cx, cy, w, h, rotation] format of shape (n, 5).
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)

def process_batch_poly(detections, labels, iouv, ignore_cls=False):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2, x3,y3, x4,y4) format.
    Arguments:
        detections (Array[N, 10]), x1, y1, x2, y2, x3,y3, x4,y4 conf, class
        labels (Array[M, 9]), class, x1, y1, x2, y2, x3,y3, x4,y4
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    # iou
    # old
    # iou = poly_iou(labels[:, 1:], detections[:, :8])
    # new
    iou = poly_iou_cpu64(labels[:, 1:].cpu().numpy().astype(np.float64), detections[:, :8].cpu().numpy().astype(np.float64))
    iou = torch.from_numpy(iou).to(labels.device)
    #labels[nlable,1+8=9]  #detections[nDetection,8+1+1=10]-->iou[nlable,nDetection]
    # dot
    # dot = vector_dot(labels[:, 1:], detections[:, :8])
    
    #在2维数据iou[nlable,nDetection]中选择True元素集合，生成匹配对集合[nid(labels)],[nid(detections)]
    #labels[:, 0:1]==detections[:, 9] labels中每一个元素与detections里面每个元素比较，相等的话就生成匹配对集合[nid(labels)],[nid(detections)]
    #以上两个集合通过 & 操作符求交集
    if not ignore_cls:
        x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 9]))  # IoU above threshold and classes match
    else:
        x = torch.where((iou >= iouv[0]))
    #x[0]=[nid(labels)],x[1]=[nid(detections)]  x是两个长度一样的tensor集成的list
    # assert(len(x)==2 and x[0].shape[0]==x[1].shape[0])
    if x[0].shape[0]:#存在匹配
        #torch.stack(x, 1)是两个list在维度为1上拼接得到[nid,2]
        #iou[x[0], x[1]]维度替换为[nid]，然后通过[:, None]扩展一维[1]-->[nid,1]
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [id_label, id_detection, iou]
        #[nid,2] cat [nid,1]得到matches[nid,3=(id_label, id_detection, iou)]
        if x[0].shape[0] > 1:#有一个以上匹配的时候需要剔除掉重复匹配，保证1对1
            if not ignore_cls:
                matches = matches[matches[:, 2].argsort()[::-1]]#按iou从大到小排序
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]#对第1列id_detection去重，因为排序了，删除同一列排序后面iou小的
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]#对第0列id_label去重，因为排序了，删除同一列排序后面iou小的
            #至此，每一行每一列都只有唯一一个1对1匹配
        matches = torch.Tensor(matches).to(iouv.device)#numpy转gpu
        #初值correct里面全部置0，下面仅在有匹配>=iouv的地方给1
        if not ignore_cls:
            correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        else:
            ass = np.unique(matches[:, 1].cpu().numpy(), return_index=True)[1].shape[0]
            bss = np.unique(matches[:, 0].cpu().numpy(), return_index=True)[1].shape[0]
            if ass == bss:
                correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
            else:
                temp = torch.ones(detections.shape[0]).bool()
                temp[matches[:,  1].long()] = False
                # temp_c = correct[matches[:,  1].long()].clone()
                temp_c = matches[:, 2:3] >= iouv
                correct = torch.cat([temp_c, correct[temp]], dim=0)
        #得到预测框detection中每个编号对应的真值框编号，有可能
    else:
        matches = torch.zeros([0,3]).to(iouv.device)
    return correct, matches#[detections.shape[0], iouv.shape[0]]