import cv2
import numpy as np
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Placeholder for the predict function

from math2.transform import A232rot_torch,xcycdudv2A23_torch,A23inverse,A23inverse_torch,apply_inverse_affine_and_map,A23Crop_torch,map_arrow
from general.config import load_config
from image.tif import read_tif_with_tifffile
from FlyCrop import FlyCrop,reload
from models.experimental import attempt_load
from detect import detect
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii, non_max_suppression,non_max_suppression_obb, non_max_suppression_dfl, \
    apply_classifier, scale_coords,scale_coords_poly, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box,xyxyxyxy2xywhr, \
    xywh2xyxy

from models.yolo import OUT_LAYER
from pts2cov import pts2cov,probiou_fly_gauss,cs2arcarrow2
from pathlib import Path
from fly.remote.model import RemoteVitModel
from fly.remote.image_crop import load_A23_file

from utils.metrics import ap_per_class, process_batch_obb
from tqdm import tqdm

from general.MyString import replace_last_path
import csv

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
    save_dir = config.get('save_dir',replace_last_path(images_dir,'crop')) #os.join(images_dir,'crop')
    image_names = [f for f in os.listdir(images_dir) if f.endswith(('.tif','.png','.jpg')) and f[0]!='#']
    conf_thres = config.get('conf_thres', 0.25)
    iou_thres = config.get('iou_thres', 0.45)
    fly_num = max(config.get('fly_num', 4), 1)
    #images = read_tif_with_tifffile(os.path.join(images_dir,image_names[0]))
    fly = FlyCrop(120.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fly.set_size(CROP_SIZE)
    # Placeholder model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iouv = torch.linspace(0.5, 0.95, 10).to(device) #0.5-->0.2

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
        stats = []
        img_num = len(image_names)
        save_dir = Path(save_dir)
        if img_num > 0:
            save_dir.mkdir(exist_ok=True)
        for index in tqdm(range(img_num)):
            globe_objects = torch.zeros(0, 10+6 + 1).to(device) #[n_total, 4*2+1+1 + 6(xyabcs) + 1]
            map = reload(os.path.join(images_dir, image_names[index]), fly.device)
            A23_GT = load_A23_file(Path(images_dir).parent / 'crop_txt' / Path(image_names[index]).with_suffix('.txt').name)
            assert A23_GT.shape[0] >= fly_num
            try:
                labels = torch.from_numpy(load_labels(Path(labels_dir)/ Path(image_names[index]).with_suffix('.txt').name)).to(device)   # pts
                labels[:, 1:] = (labels[:, 1:].reshape(-1, 4, 2) * torch.as_tensor([[map[0].shape[1], map[0].shape[0]]], device=device)).reshape(-1, 8)
            except:
                labels = torch.zeros((0, 9), dtype=torch.float32).to(device)
            map_height, map_width = map[0].shape[0], map[0].shape[1]
            fly.link_map(map)  # Replace with your large map image path
            labels_mask = torch.zeros(labels.shape[0], dtype=bool, device=device)
            for i in range(0, ((A23_GT.shape[0]-1) // fly_num) + 1):
                image = None
                cur_cams = []
                pre_cams = []
                try:
                    for j in range(fly_num):
                        fly.set_affine(A23_GT[i * fly_num + j])
                        image = fly.render() if image is None else torch.cat([image, fly.render()], dim=0) #image[B,C,H,W]
                        A23t = fly.mA.squeeze(0)
                        t_cos, t_sin, t_s, t_xc, t_yc = A232rot_torch(A23t, CROP_SIZE)
                        cur_cams.append((t_xc, t_yc, t_cos, t_sin, A23t))
                        # A23t_inv = A23inverse_torch(A23t)
                        if labels.shape[0] > 0:
                            gt_pts = labels[:, 1:].clone().reshape(-1, 4, 2)
                            gt_pts = torch.cat([gt_pts, torch.ones_like(gt_pts[:, :, :1])], dim=-1)
                            gt_pts = torch.einsum('ijk, km->ijm', gt_pts, A23t.T).mean(1)   # nt, 4, 2 --> nt, 2
                            labels_mask |= ((gt_pts >= torch.as_tensor([[0, 0]], device=device)) & (gt_pts <= torch.as_tensor([CROP_SIZE], device=device))).all(-1)
                except IndexError:
                    pass

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
                    image_pred = fly.render(pre_cams[idx][-1][None])[0].permute(1, 2, 0).contiguous().to(device='cpu').numpy()
                    cv2.imencode('.jpg', image_show)[1].tofile(f'{save_dir / Path(image_names[index]).stem}_{i * fly_num + idx}_gt.jpg')
                    cv2.imencode('.jpg', image_pred)[1].tofile(f'{save_dir / Path(image_names[index]).stem}_{i * fly_num + idx}_pred.jpg')
                    if len(det): #det[np,4(pts)*2+1(conf)+1(cls)]
                        # Rescale boxes from img_size to image_show size
                        det[:, :8] = scale_coords_poly(img.shape[2:], det[:, :8], image_show.shape).round()
                        # Print results
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
                            globe_objects = globe_objects[~iou_cls_mask] #globe_objects[np_globe,4(pts)*2+1(conf)+1(cls) + 6(xyabcs) + 1]
                        
                        obj17 = torch.cat([det_new,det_cov,det_side], dim=1) #obj17[np,17=10(4*2+1(conf)+1(cls))+6(xyabcs) + 1(side)]
                        globe_objects = torch.cat([obj17, globe_objects], dim=0) #globe_objects[ntotal,17=10(4*2+1(conf)+1(cls))+6(xyabcs) + 1(side)]


            if labels.shape[0] > 0:
                labels_filter = labels[labels_mask]
                if labels_filter.shape[0] > 0:
                    if globe_objects.shape[0] > 0:
                        globe_objects_xywhr = torch.cat([xyxyxyxy2xywhr(globe_objects[:, :8]), globe_objects[:, 8:10]], dim=-1)
                        correct = process_batch_obb(globe_objects_xywhr, labels_filter, iouv)
                        stats.append([correct.cpu().numpy(), globe_objects[:, 8].cpu().numpy(), globe_objects[:, 9].cpu().numpy(), labels_filter[:, 0].cpu().numpy()])
                    else:
                        stats.append((torch.zeros(0, iouv.shape[0], dtype=torch.bool), torch.Tensor(), torch.Tensor(), labels_filter[:, 0].cpu().numpy()))

        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        map_text = []
        p, r, ap, f1, ap_class, threshs, py = ap_per_class(*stats, plot=False, names=dict_name)
        nt = np.bincount(stats[3].astype(np.int64), minlength=len(names))
        ap50, ap595 = ap[:, 0], ap.mean(1)
        map50, map595 = ap50.mean(), ap.mean()
        map_text.append(('%18s' + '%11s' * 2 + '%11s' * 4) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95'))
        pf = '%18s' + '%11i' * 2 + '%11.4g' * 4
        map_text.append(pf % ('All', len(image_names), nt.sum(), p.mean(), r.mean(), map50, map595))
        for i, idx in enumerate(ap_class):
            map_text.append(pf % (names[idx], len(image_names), nt[idx], p[i], r[i], ap50[i], ap595[i]))
        print('\n'.join(map_text))
        

        with open(save_dir.parent / 'classes_map.csv', 'w', newline='') as file_map:
            writer = csv.writer(file_map)
            writer.writerow(['Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95'])
            writer.writerow([f'{a:.6f}' if not isinstance(a, str) else a for a in ['All', len(image_names), nt.sum(), p.mean(), r.mean(), map50, map595] ])
            # Print results per class
            for i, idx in enumerate(ap_class):
                writer.writerow([f'{a:.6f}' if not isinstance(a, str) else a for a in [names[idx], len(image_names), nt[idx], p[i], r[i], ap50[i], ap595[i]]])


    sys.exit()
