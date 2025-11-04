# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread
import cv2

import numpy as np
import torch
from tqdm import tqdm
from utils.metrics import process_batch_poly
from general.MyString import add_suffix_to_filename
from models.yolo import OUT_LAYER

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.callbacks import Callbacks
from utils.datasets import create_bigimg_dataloader, img2pts_paths
from utils.general_nms import (LOGGER, check_file, check_requirements, big_nms,check_dataset, increment_path,  print_args)
from utils.general import (check_img_size,xyxyxyxy2xywhr,xywh2xyxy,colorstr)
from utils.metrics import ap_per_class, ConfusionMatrix, process_batch, process_batch_obb
from utils.torch_utils import select_device, time_sync,torch_distributed_zero_first
from tools.plotbox import plot_one_box,plot_one_rot_box
from detect_big import detect_big
import cv2
import csv # save classes_map to excel

import random

import pickle

def filter_images(path, extensions):
    # 获取指定路径下所有文件
    all_files = sorted(os.listdir(path))
    # 根据扩展名过滤文件
    image_files = []
    for file in all_files:
        if any(file.lower().endswith(ext.lower()) for ext in extensions.split(';')):
            image_files.append(os.path.join(path, file)) 
    return image_files
def load_files(path, extensions, test_count=None):
    if(os.path.exists(path)):
        files = sorted(os.listdir(path))
        image_files = filter_images(path, extensions)
        # 如果指定了 test_count，且小于文件总数，则随机选择 test_count 张图片
        if test_count is not None and test_count < len(image_files) and test_count > 0:
            image_files = random.sample(image_files, test_count)

        label_files = img2pts_paths(image_files)
        return image_files, label_files
    else:
        return [],[]


def read_image_label(detect_mode, im_file, pts_file):#参考dataset.py里面的verify_image_label
    image = cv2.imdecode(np.fromfile(im_file, dtype=np.uint8),cv2.IMREAD_COLOR)#cv2.imread(im_file)

    base_name, extension = os.path.splitext(pts_file)
    lb_file = base_name+'.txt'
    if os.path.exists(lb_file):
        with open(lb_file) as f:
            l = [x.split() for x in f.read().strip().splitlines() if len(x)]
            read_l = len(l)
            l = np.array(l, dtype=np.float32)#[nt,5]
            assert(l.shape[0]==read_l)
    else:
        l = np.empty((0,5), dtype=np.float32)#[nt,5]
    
    if os.path.exists(pts_file):
        with open(pts_file) as f:
            p = [x.split() for x in f.read().strip().splitlines() if len(x)]
            if len(p):#参考dataset.py里面的verify_image_label
                for i in range(len(p)):
                    if p[i]==['-']:
                        xc,yc,w,h = float(l[i][1]),float(l[i][2]),float(l[i][3]),float(l[i][4])
                        #p[i] = [str(round(xc-w/2,6)),'0','0','0','0','0','0','0']
                        p[i] = [str(round(xc-w/2,6)),str(round(yc-h/2,6)),
                                str(round(xc+w/2,6)),str(round(yc-h/2,6)),
                                str(round(xc+w/2,6)),str(round(yc+h/2,6)),
                                str(round(xc-w/2,6)),str(round(yc+h/2,6))]
                p = np.array(p, dtype=np.float32)#list转np矩阵[nt,4*2=8]
    else:
        p = np.empty((0,8), dtype=np.float32)#list转np矩阵[nt,4*2=8]

    nl = len(l)
    if nl:
        clss = l[:, 0].reshape(-1, 1)#[nt,1]，我感觉reshape(-1, 1)不必要，本来l[:, 0]就是在dim=1就只有长度1
        boxs = np.clip(l[:, 1:], 0, 1)#[nt,4] ,np.clip的意思是设定上下限,目标框的坐标和wh都限制在0,1范围内
        l = np.concatenate((clss, boxs), axis=1)#l从list转换成np矩阵[nt,1+4=5]
        if len(p)>0:
            assert nl==len(p)
            # p = np.clip(p, 0, 1)
            l = np.concatenate((l, p), axis=1)#l[l(5)+p(4*2)]
            assert l.shape[1] == 13, f'labels require 5+8=13 columns, {l.shape[1]} columns detected'
        else: #l[nt,5]
            assert l.shape[1] == 5, f'labels require 5=1+4 columns, {l.shape[1]} columns detected'
        #assert (l >= 0).all(), f'negative label values {l[l < 0]}'
        #assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
        margin = 0.3
        l = l[(l[:, 0] >= 0) & (l[:, 1:]>-margin).all(-1) & (l[:, 1:]<(1+margin)).all(-1)]
        assert (l[:, 0] >= 0).all(), f'negative label values {l[l[:, 0] < 0]}'
        assert ((l[:, 1:] > -margin) & (l[:, 1:] < (1+margin))).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > (1+margin) | l[:, 1:] < -margin]}'
        l = np.unique(l, axis=0)  # remove duplicate rows,  同一目标重复标注，合并
        if len(l) < nl:
                #segments = np.unique(segments, axis=0)
                msg = f'WARNING: {im_file}: {nl - len(l)} duplicate labels removed'
                print(msg)
    else:
        if len(p)>0:
            l = np.zeros((0, 1+4 + 4*2), dtype=np.float32)
        else:
            l = np.zeros((0, 1+4), dtype=np.float32)
    return image, l#[nt, 4+1 + 4*2]
def find_right_value(sz, subsizes, start=0):
    for i in range(start, len(subsizes)-1):
        if subsizes[i] < sz < subsizes[i+1]:
            return subsizes[i+1]
    return subsizes[len(subsizes)-1]

import numpy as np
import cv2
def calculate_area_ratio(keeps, width, height,inside_rate=0.5):
    # 使用OpenCV计算目标在图像内部区域的面积比。
    # :param x: 目标的四个顶点坐标，形状为(n, 8)。
    # :param width: 图像宽度。
    # :param height: 图像高度。
    # :return: 更新后的目标顶点坐标。
    n = len(keeps)
    indices_to_keep = []
    image_poly = np.array([[0,0],[width,0],[width,height],[0,height]]).astype(np.float32)
    #image_mask = np.zeros((height, width), dtype=np.uint8)

    for i,obj in enumerate(keeps):
        # 从数组中提取顶点并形成多边形
        contour = obj[:8].reshape(-1, 2).cpu().numpy().astype(np.float32)
        original_area = cv2.contourArea(contour)

        intersection_area, intersection = cv2.intersectConvexConvex(contour, image_poly)

        # 计算面积比
        area_ratio = intersection_area / original_area

        # 如果面积比大于等于0.7，则保留
        if area_ratio >= inside_rate:
            indices_to_keep.append(i)

    # 返回过滤后的数据
    return indices_to_keep
@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=[640,640],  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        ab_thres=3.0,
        iou_thres=0.6,  # 0.1 NMS IoU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        plots_count=4,
        plots_mask=0,
        callbacks=Callbacks(),
        fold=2,
        subsizes=[256,512,768,1024],
        overlap=100,
        multi_label=0,
        sub_size_scale=2.5,
        resize=1
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt = next(model.parameters()).device, True  # get model device, PyTorch model

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # model = DetectMultiBackend(weights, device=device, dnn=False)
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride, pt = model.stride, True
        stride_max = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=stride_max)  # check image size
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        model.model.half() if half else model.model.float()
        
        # Data
        data = check_dataset(data)  # check
    
    # 判断模型类别
    pts, dfl_flag = False, False
    for mname in OUT_LAYER.keys():
        m = model.get_module_byname(mname)
        if m is not None:
            pts = mname in ['DetectROT', 'OBB']
            dfl_flag = mname in ['DetectDFL', 'OBB']
            break
    mname = OUT_LAYER[mname]
    _process_batch = process_batch_obb if pts else process_batch

    # Configure
    model.eval()
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and device.type != 'cpu':
            imgsz2 = [imgsz,imgsz] if isinstance(imgsz,int) else imgsz
            model(torch.zeros(1, 6, imgsz2[0], imgsz2[1]).to(device).type_as(next(model.model.parameters())))  # warmup
        

    seen = 0.00001
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    s = ('%20s' + '%11s' * 8) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'f1', 'thresh')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(6, device=device)
    jdict, stats, ap, ap_class = [], [], [], []

    train_path = Path(weights).parent
    threshs_file_name = train_path / 'threshs.npy'
    if(plots_mask and os.path.exists(threshs_file_name)):
        threshs = np.load(threshs_file_name)
        assert threshs.shape[0]==len(names)
        conf_thres = min(threshs[threshs>0])
    else:
        threshs = conf_thres * np.ones(len(names))
    assert len(threshs)==len(names)

    # colors
    colors = [
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
    data_dict = check_dataset(data)  # check if None
    val_count = data_dict.get('val_count',0)
    random.seed(0)
    val_big_path = os.path.join(data_dict.get('path',''), data_dict.get('val_big',''))
    image_files, label_files = load_files(val_big_path,'.jpg;.bmp;.tif;.png',val_count)#8000
    assert len(image_files)==len(label_files)
    if len(image_files)==0:
        print(f'\033[91m{val_big_path} not exists or image_files[{len(image_files)}]\033[0m')
    for i,(image_path, label_path) in enumerate(tqdm(zip(image_files, label_files), desc=s, total=len(image_files))):
        #print(i,image_path)
        t1 = time_sync()
        try:
            image, label = read_image_label("Detect_pts", image_path, label_path)#参考dataset.py里面的verify_image_label
            if image is None:
                print(f'\033[91m{image_path} Can not read.\033[0m')
                continue
        except Exception as e:
            print(f'\033[91m{image_path} Un known error.\033[0m')
            continue
        assert len(image.shape)==3
        height, width = image.shape[:2]
        
        subsize=2*[int]
        subsize[0] = find_right_value(height,subsizes)
        subsize[1] = find_right_value(width,subsizes)
        if sub_size_scale > 0:
            imgsz[0] = int(sub_size_scale * subsize[0])
            imgsz[1] = int(sub_size_scale * subsize[1])
        else:
            imgsz[0] = subsize[0]
            imgsz[1] = subsize[1]

        # if(image.shape[0] < subsize):
        #     image = cv2.resize(image, (imgsz,imgsz))
        #image[H,W, C]   label[nt, 13 = 1(cls)+4(box) + 4*2]
        p = Path(image_path)
        save_path = str(save_dir / p.name)
        t2 = time_sync()
        dt[0] += t2 - t1
        #
        keeps, dataloader = detect_big(model,half,device, image, imgsz, batch_size,subsize,overlap, conf_thres,iou_thres,single_cls = multi_label,mname=mname,xyoff=[0,0],resize=resize)
        ids = calculate_area_ratio(keeps, width, height, 0.5)
        assert len(ids)<=len(keeps)
        keeps = [keeps[i] for i in ids]
        dt[1] += time_sync() - t2
        seen += 1
        # plot
        # for k in keeps:
        #     xyxy = k[:8].cpu().numpy()
        #     cls = k[-1].cpu()
        #     plot_one_rot_box(xyxy, image, color=colors[int(cls)],   dir_line=True)
        # cv2.imwrite(save_path, image)
        assert label.shape[-1]==1+4 or label.shape[-1]==1+4+8
        if label.shape[-1]==1+4+8: #obb
            pts_size = label.shape[-1] - 5
            targets = torch.zeros((label.shape[0],1+pts_size),device=device)
            targets[:, 0] = torch.from_numpy(label[:, 0])  # 拷贝cls部分
            targets[:, 1:] = torch.from_numpy(label[:, 1+4:])  # 拷贝除了框4(xywh)之外的部分
            #targets = torch.from_numpy(label)#targets[4+1 + 4*2] label[nt, 4+1 + 4*2]
            #targets = targets[..., 4:]#把前面的4(box)删掉了！targets[1(cls) + 4*2]
            #targets = targets.to(device)#targets[1(cls) + 4*2]
            targets[:, 1:] *= torch.Tensor([width, height, width, height, width, height, width, height]).to(device)  # to pixels
            if len(keeps):#keeps[nt][10 = 4(pts)*2+1(conf)+1(cls)]-->pred[nt,10 = 4(pts)*2+1(conf)+1(cls)]
                pred = torch.stack(keeps, dim=0)
                assert pred.shape[-1]==8+2
            else:
                pred = torch.zeros(0,10,device=device)#for no object case, same dim
            assert pred.shape[-1]==10
        else: #hbb
            targets = torch.from_numpy(label).to(device) #targets[nt,5=1(cls)+4(xywh)]
            targets[:,1:] = xywh2xyxy(targets[:,1:]) #targets[nt,5=1(cls)+4(xywh)]->targets[nt,5=1(cls)+4(xyxy)]
            targets[:, 1:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            if len(keeps):#keeps[nt][10 = 4(pts)*2+1(conf)+1(cls)]-->pred[nt,10 = 4(pts)*2+1(conf)+1(cls)]
                pred = torch.stack(keeps, dim=0) #pred[np,6=4(xyxy)+1(conf)+1(cls)]
                assert pred.shape[-1]==4+2
            else:
                pred = torch.zeros(0,6,device=device)#for no object case, same dim
            assert pred.shape[-1]==6
        #targets[1(cls)+4(pts)*2]  Detect:[ngt, 1(cls)+4(pts)*2]
        #pred[10 = 4(pts)*2+1(conf)+1(cls)]  Detect:[nt, 10 = 4(pts)*2+1(conf)+1(cls)]
        nl = len(targets)
        tcls = targets[:, 0].tolist() if nl else [] # tcls[nt] target class
        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue
        if nl:
            assert pred.shape[-1]==10 or pred.shape[-1]==6
            conf_cls = pred[:, -2:]
            #
            if pred.shape[-1]==10:
                pbox = pred[:,:8]
                pred[:, :5] = xyxyxyxy2xywhr(pbox)
                pred = torch.cat((pred[:,:5], conf_cls), 1)
            else:
                pbox = pred[:,:4]
                pred = torch.cat((pbox, conf_cls), 1)

            correct = _process_batch(pred, targets, iouv) #correct[npred,nt]
                
            # if plots:
            #     confusion_matrix.process_batch(pred, labelsn)
        else:
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        #
        stats.append((correct.cpu(), pred[:, -2].cpu(), pred[:, -1].cpu(), tcls))  # (correct, conf, pcls, tcls)
        
        #only in Detect but not in Detect_pts
        if plots_count>0:
            if(len(keeps)>0):#有目标才存标记图
                for k in keeps:#keeps[nt][10=4(pts)*2+1(conf)+1(cls) / 6=4(xyxy)+1(conf)+1(cls)]
                    pts = k[:-2].cpu().numpy()#k[4(pts)*2+1(conf)+1(cls)] / k[6=4(xyxy)+1(conf)+1(cls)]
                    conf = k[-2].cpu()
                    if conf>0.25:
                        cls = k[-1].cpu()
                        label = '{} {:.2f}'.format(names[int(cls)], conf)
                        if pts.shape[0]==8:
                            plot_one_rot_box(pts, image, color=colors[int(cls)%len(colors)], label=label, dir_line=True)
                        else:
                            assert pts.shape[0]==4
                            plot_one_box(pts, image, color=colors[int(cls)%len(colors)], label=label, line_thickness=None)
                save_path2 = str(save_dir / p.with_suffix('.jpg').name)
                cv2.imencode('.jpg', image)[1].tofile(save_path2)#cv2.imwrite(save_path, image)
                plots_count-=1
        
    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class, threshs, py = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        #p[nc] r[nc] ap[nc,10] f1[nc] ap_class[nc] threshs[nc]
        # 保存到文件
        with open(save_dir / 'status.pkl', 'wb') as f:
            pickle.dump([py,ap,names], f)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        train_path = weights.parent.parent if not training and not isinstance(weights, str) else save_dir
        np.save(train_path / 'threshs.npy', threshs)
    else:
        nt = torch.zeros(1)
        f1 = torch.zeros(1)
        threshs = torch.zeros(1)
    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.4g' * 6  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map, f1.mean(), threshs.mean()))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], f1[i], threshs[i]))

    train_path = save_dir
    with open(train_path / 'classes_map.csv', 'w', newline='') as file_map:
        writer = csv.writer(file_map)
        writer.writerow(['Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'f1', 'thresh'])
        writer.writerow([f'{a:.6f}' if not isinstance(a, str) else a for a in ["all", seen, nt.sum(), mp, mr, map50, map, f1.mean(), threshs.mean()] ])
        # Print results per class
        for i, c in enumerate(ap_class):
            writer.writerow([f'{a:.6f}' if not isinstance(a, str) else a for a in [names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], f1[i], threshs[i]]])
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        shape = (batch_size, 3, imgsz, imgsz)
        writer.writerow([f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t])
        
    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        imgsz2 = [imgsz,imgsz] if isinstance(imgsz,int) else imgsz
        shape = (batch_size, 3, imgsz2[0], imgsz2[1])
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference sub img at shape {shape}, %.1fms NMS per big image' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/Guge.yaml', help='dataset.yaml path')###
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp130/weights/best.pt', help='model.pt path(s)')###
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')###
    parser.add_argument('--imgsz', '--img', '--img-size', type=list, default=[1280,1280], help='inference size (pixels)')###
    parser.add_argument('--conf_thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='NMS IoU threshold')###
    parser.add_argument('--ab_thres', type=float, default=3.0, help='a b thres')###
    parser.add_argument('--fold', type=int, default=2, help='倍角')#
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--subsizes', type=list, default=[640,1024,1280], help='subsizes')
    parser.add_argument('--overlap', type=int, default=80, help='overlap')
    parser.add_argument('--multi_label', type=int, default=0, help='multi_label')
    parser.add_argument('--sub_size_scale', type=float, default=1.0, help='sub_size_scale')
    parser.add_argument('--plots_mask', type=int, default=0, help='plots_mask')
    parser.add_argument('--resize', type=int, default=0, help='resize')
    opt = parser.parse_args()
    #opt.data = 'data/Guge.yaml'
    #opt.weights = 'runs/train/exp130/weights/best.pt'
    #opt.batch_size = 16
    #opt.conf_thres = 0.001
    #opt.iou_thres = 0.6
    #opt.ab_thres = 3.0
    #opt.fold = 2

    #dota1.5 #62.01-48.1
    opt.data = 'data/dota.yaml'
    opt.weights = 'runs/train/s-obb-79.52-61.01-use_ioa-False-4090/weights/best.pt'
    # opt.subsizes = [640,1024,1280]
    opt.batch_size = 6
    #coco2017 #55.62-39.33
    # opt.data = 'data/coco2017.yaml'
    # opt.weights = '../../yolov5-dfl/s-62.15-43.63-v11s-dfl/weights/best.pt'
    # # opt.subsizes = [640,1024,1280]
    # opt.batch_size = 6

    opt.iou_thres = 0.45
    opt.multi_label = 0
    opt.sub_size_scale = 1.0 # imgsz / sub_size_scale = subsize
    #opt.plots_mask = 1
    #Guge
    # opt.data = 'data/Guge.yaml'
    # opt.weights = 'runs/train/Guge/exp_Guge10/weights/best.pt'
    # opt.batch_size = 16
    # opt.imgsz = 768
    # opt.conf_thres = 0.001
    # opt.iou_thres = 0.1
    # opt.ab_thres = 3.0
    # opt.subsize = opt.imgsz

    # opt.data = ROOT / 'data/dota43516-7878.yaml'
    # opt.weights = ROOT / 'runs/train/dota43516-7878/exp_7473-pts-l/weights/best.pt'
    # opt.imgsz = 640
    # opt.conf_thres = 0.001
    # opt.iou_thres = 0.15
    # opt.subsize = 480
    # opt.overlap = 100

    print_args(FILE.stem, opt)
    return opt


def main(opt):

    if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
        LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(opt.data)  # check if None

    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
