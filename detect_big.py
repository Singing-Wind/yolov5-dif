# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from utils.datasets import create_bigimg_dataloader
from detect import detect
from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadBigImages
from utils.general_nms import (LOGGER, check_file, check_requirements, big_nms,check_dataset, increment_path,  print_args)
from utils.general import (scale_coords_poly, apply_affine_transform)
from utils.torch_utils import select_device, time_sync, torch_distributed_zero_first
from tools.plotbox import plot_one_box,plot_one_rot_box
# from DOTA_devkit.ResultMerge_multi_process import py_cpu_nms_poly_fast
from models.yolo import OUT_LAYER
from img2word.img2word import extract_object_texts

def filter_targets(det, image_shape, expand=0):
    # 过滤掉中心点在图像范围以外的目标
    # :param det: 包含目标坐标的数组，形状为 (n, 8)
    # :param image_shape: 图像的形状 (height, width)
    # :return: 过滤后的目标数组
    img_height, img_width = image_shape[:2]   
    # 计算每个目标的中心点坐标
    if det.shape[1]==8:
        x_center = det[:, [0, 2, 4, 6]].mean(dim=1)
        y_center = det[:, [1, 3, 5, 7]].mean(dim=1)
    else:
        x_center = det[:, 0]
        y_center = det[:, 1]
    # 创建一个布尔掩码，判断中心点是否在图像范围内
    mask = (x_center >= -expand) & (x_center < img_width+expand) & (y_center >= -expand) & (y_center < img_height+expand)
    # 过滤目标
    filtered_det = det[mask]
    return filtered_det
def detect_big(model,half,device, image, imgsz, batch_size,subsize,overlap, conf_thres,iou_thres,single_cls=False,mname=2,xyoff=[0,0],resize=0):
    if resize:
        if(image.shape[0] < subsize[0] or image.shape[1] < subsize[1]):
            if image.shape[0] < subsize[0] and image.shape[1] < subsize[1]:
                new_img = np.zeros((subsize[0],subsize[1],3),dtype=np.uint8)#下右拓展
            elif image.shape[0] < subsize[0]:#下拓展
                new_img = np.zeros((subsize[0],image.shape[1],3),dtype=np.uint8)
            elif image.shape[1] < subsize[1]:#右拓展
                new_img = np.zeros((image.shape[0],subsize[1],3),dtype=np.uint8)
            new_img += 114
            new_img[:image.shape[0],:image.shape[1]] = image
            image = new_img
            # cv2.imshow("exp-image",image)
            # cv2.waitKey(0)
    dataloader = create_bigimg_dataloader(image, imgsz, batch_size, subsize=subsize, over_lap=overlap,xyoff=xyoff, resize=resize, big_warp=1, swaprgb=0)
    det_results = {}
    shape_img = (int(imgsz[0]), int(imgsz[1])) #shape_img[H,W]
    for xys,ims in dataloader: #ims[B,3,H,W], xys[B,2(A,A_1),2,3]
        assert ims.shape[2:]==shape_img
        ims = ims.to(device)
        ims = ims.half() if half else ims.float()  # uint8 to fp16/32
        ims /= 255  # 0 - 255 to 0.0 - 1.0
        if len(ims.shape) == 3:
            ims = ims[None]  # expand for batch dim
        # Inference
        # pred = detect(model, False, ims, False,conf_thres, iou_thres, ab_thres, fold_angle=fold, mask_dir=mask_dir,threshs=threshs,multi_label=multi_label)
        pred = detect(model, ims, False, conf_thres, iou_thres, mname=mname,agnostic_nms=single_cls,classes=None,max_det=3000)
        #pred[b][nt,10=4(pts)*2+1(conf)+1(cls)]

        for i, det in enumerate(pred):
            if len(det):
                # xy = torch.tensor([[xy[0], xy[1]]])
                if resize==0:
                    A23_1 = xys[i,1] #xys[B,2,2,3]->xy[2,3]
                    #通过逆阵A23_1还原得到det[:, :8]大图绝对坐标
                    det = apply_affine_transform(det, A23_1.to(device))
                    #过滤逆变换大图出界的目标
                    det = filter_targets(det, image.shape, 8)                        
                else:
                    assert det.shape[-1]==6 or det.shape[-1]==10
                    pts_size = det.shape[-1] - 2 #[1(conf)+1(cls)]
                    xy = xys[i]
                    xy = xy.repeat(1, pts_size//2)#pts_size/2个点
                    xy = xy.to(device)
                    det[:, :pts_size] = (xy + scale_coords_poly(ims.shape[2:], det[:, :pts_size], tuple(subsize))).round()
                    #det[:, :8]除以im.shape[2:]再乘以cimg.shape[1:]加上左上角坐标加上左上角坐标xy得到大图绝对坐标
                
                for i, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    item = {
                        'xyxy': list(xyxy),
                        'conf': conf
                    }
                    items = det_results.setdefault(int(cls), [])
                    items.append(item)
    # 大图nms
    keeps = []
    for cls, dets in det_results.items():
        det_list = []
        scores = []
        for item in dets:
            det_list.append(item['xyxy'])
            scores.append(item['conf'])
        det_list = torch.tensor(det_list).to(device)
        scores = torch.tensor(scores).to(device)
        assert det_list.shape[0]==scores.shape[0]
        if det_list.shape[0]>0: #det_list[np,8/4] scores[np]
            indexes = big_nms(det_list, scores, iou_thres)
            cls = torch.tensor([cls]).to(device)
            for idx in indexes:
                xyxy = det_list[idx]
                keeps.append(torch.cat((xyxy, scores[idx][None], cls), dim=0))
    return keeps,dataloader#keeps[nt][4(pts)*2+1(conf)+1(cls)]

def find_right_value(sz, subsizes,start=0):
    for i in range(start, len(subsizes)-1):
        if subsizes[i] < sz < subsizes[i+1]:
            return subsizes[i+1]
    return subsizes[len(subsizes)-1]

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=[640,640],  # inference size (pixels)
        conf_thres=0.1,  # confidence threshold
        iou_thres=0.1,  # NMS IOU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=False,  # save results to *.txt
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        half=True,  # use FP16 half-precision inference
        plot_label=True,
        dir_line=True,
        subsize=[512,512],
        overlap=100,
        data='',
        multi_label=0,
        sub_size_scale=2.5,
        subsizes=[256,512,768,1024],
        resize=0,
        xyoff=[0.0,0.0],
        url = '' #'http://10.255.255.1:8000/v1/chat/completions'
        ):
    source = str(source)
    save_img = True
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    if is_url and is_file:
        source = check_file(source)  # download
    if source=='' or not os.path.exists(source):
        source0 = source
        data = check_dataset(data)
        source = os.path.join(data.get('path',''), data.get('val_big',''))
        print(f'\033[91mdata path:[{source0}] not exists. change to [{source}].\033[0m')

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    print(f'\033[92m{save_dir}\033[0m')

    # Load model
    device = select_device(device)
    # model = DetectMultiBackend(weights, device=device, dnn=False)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    stride, names = model.stride, model.names
    stride_max = max(int(model.stride.max()), 32)  # grid size (max stride)

    # Half
    pt = True
    if pt:
        model.model.half() if half else model.model.float()
    # Dataloader

    dataset = LoadBigImages(source, img_size=imgsz, stride=stride_max, auto=False, subsize=[0,0], over_lap=overlap)
    bs = 1  # batch_size

    train_path = Path(weights).parent
    if(os.path.exists(train_path / 'threshs.npy')):
        threshs = np.load(train_path / 'threshs.npy')
        assert threshs.shape[0]==len(names)
        conf_thres = min(threshs[threshs>0])
    else:
        threshs = conf_thres * np.ones(len(names))
    
    # 判断模型类别
    pts, dfl_flag = False, False
    for mname in OUT_LAYER.keys():
        m = model.get_module_byname(mname)
        if m is not None:
            pts = mname in ['DetectROT', 'OBB']
            dfl_flag = mname in ['DetectDFL', 'OBB']
            break
    mname = OUT_LAYER[mname]

    # Run inference
    # if pt and device.type != 'cpu':
    #     model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
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

    # for path, im, im0s, vid_cap, s in dataset:
    for path, cut_imgs, convert_imgs, image,  cmt in dataset:#利用dataset切割管理器，取出每块信息，LoadBigImages::__next__
        # print(path)
        s = os.path.basename(path)
        height, width = image.shape[:2]
        
        subsize=2*[int]
        subsize[0] = find_right_value(height,subsizes)
        subsize[1] = find_right_value(width,subsizes)
        if sub_size_scale > 0:
            imgsz[0] = sub_size_scale * subsize[0]
            imgsz[1] = sub_size_scale * subsize[1]
        else:
            imgsz[0] = subsize[0]
            imgsz[1] = subsize[1]

        p = path
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg
        txt_path = str(save_dir / 'labels' / p.stem)
        if not os.path.exists(str(save_dir / 'labels')):
            os.makedirs(str(save_dir / 'labels'))
        s += '%gx%g ' % image.shape[:2]
        det_results = {}
        t1 = time_sync()
        batch_size = 1
        t0 = time_sync()
        keeps, dataloader = detect_big(model,half,device, image, imgsz, batch_size,subsize,overlap, threshs, iou_thres,single_cls = multi_label,mname=mname,xyoff=[0,0],resize=resize)
        #keeps[nt][4(pts)*2+1(conf)+1(cls)]
        if url!='':
            texts = extract_object_texts(image,keeps,names,Chinese='', url = url) #texts[nt]  ' in chinese'
            assert len(texts)==len(keeps)
        else:
            texts = None
        t3 = time_sync()
        # plot
        if(len(keeps)>0):#有目标才存标记图
            for i,k in enumerate(keeps):#keeps[nt][10=4(pts)*2+1(conf)+1(cls) / 6=4(xyxy)+1(conf)+1(cls)]
                assert k.shape[-1]==6 or k.shape[-1]==10
                text = texts[i] if texts is not None else None
                cls = k[-1].cpu()
                conf = k[-2].cpu()
                label = '{} {:.2f}'.format(names[int(cls)], conf) if plot_label else None
                if k.shape[-1]==10:
                    pts = k[:8].cpu().numpy()#k[4(pts)*2+1(conf)+1(cls)]
                    plot_one_rot_box(pts, image, color=colors[int(cls)%len(colors)], label=label, dir_line=True,text=text)
                else:
                    assert k.shape[-1]==6
                    rect = k[:4].cpu().numpy()#k[6=4(xyxy)+1(conf)+1(cls)]
                    plot_one_box(rect, image, color=colors[int(cls)%len(colors)], label=label,text=text)
            save_path2 = str(save_dir / p.with_suffix('.jpg').name)
            cv2.imencode('.jpg', image)[1].tofile(save_path2)#cv2.imwrite(save_path, image)

        # # Print time (inference-only)
        LOGGER.info(f'{s}Done. {len(keeps)}objs ({t3 - t0:.3f}s)')




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=list, default=[640, 640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--subsizes', type=list, default=[640,1024,1280], help='subsizes')
    parser.add_argument('--plot_label',action='store_true', default=True, help='plot labels')
    parser.add_argument('--subsize', type=list, default=[512,512], help='subsize')
    parser.add_argument('--overlap', type=int, default=100, help='overlap')
    parser.add_argument('--multi_label', type=int, default=0, help='multi_label')
    parser.add_argument('--sub_size_scale', type=float, default=2.5, help='sub_size_scale')
    parser.add_argument('--resize', type=int, default=0, help='resize')
    parser.add_argument('--url', type=str, default='http://10.255.255.1:8000/v1/chat/completions', help='resize') #'http://10.255.255.1:8000/v1/chat/completions'
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)

    #dota
    # opt.data = 'data/dota.yaml'
    # opt.weights = 'runs/train/dota/exp_dota2/weights/best.pt'
    # opt.imgsz = [896,896]
    # opt.subsize = opt.imgsz
    # opt.source = '/media/liu/f4854541-32b0-4d00-84a6-13d3a5dd30f2/datas/GuGe/images'
    #TZ-car-det
    # opt.data = 'data/TZ-car_det.yaml'
    # opt.weights = r'D:\yolov5\yolov5-rot\yolov5-rot-sub_size_auto2\runs\train\exp_TZ-car_det16/weights/last.pt'
    # opt.imgsz = [1280,1280] #[640,640]
    # opt.subsize = [512,512] #[256,256]   #opt.imgsz
    opt.iou_thres = 0.45
    opt.sub_size_scale = 2.5
    # #CatBug
    # opt.source = 'H:/datas/car_det_train/images-test'
    opt.resize = 0
    # opt.multi_label=0
    #4090-2
    #opt.source = '/media/liu/f4854541-32b0-4d00-84a6-13d3a5dd30f2/datas/car_det_train/TZ-test/images'
    #Guge
    # opt.data = 'data/Guge.yaml'
    # opt.weights = 'runs/train/Guge/exp_Guge10/weights/best.pt'
    # opt.source = '/home/LIESMARS/2019286190105/datasets/final-master/UCASALL/UCAS/val/images'
    # opt.source = '/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA512/val-big/images'
    # opt.source = '/home/LIESMARS/2019286190105/datasets/final-master/UCASALL/UCAS_split/val/images'
    # opt.source = '/home/LIESMARS/2019286190105/datasets/final-master/UCASALL/UCAS/val/images'
    # opt.source = 'E:/datas/GuGe/images'
    # opt.conf_thres = 0.25
    # opt.iou_thres = 0.45
    # opt.imgsz = [640, 640]
    #Guge
    # opt.weights = 'runs/train/hrsc2016/exp_hrsc2016/weights/last.pt'
    # opt.source = 'E:/datas/HRSC2016/test/images'
    # opt.conf_thres = 0.25
    # opt.imgsz = [640,832]
    #dota
    opt.weights = '../models/s-obb-80.19-61.04-dfl=0/weights/best.pt'
    opt.source = '' #"/data/datas/dota1.5/images"
    if not os.path.exists(opt.source):
        data = 'data/dota.yaml'
        data_dict = check_dataset(data)  # check if None
        opt.source = os.path.join(data_dict['path'],data_dict['val_big'])
        assert(os.path.exists(opt.source))

    #dfl coco2017
    # opt.weights = '../../yolov5-dfl/s-62.15-43.63-v11s-dfl/weights/best.pt'
    # opt.source = 'val/patches/images'
    # if not os.path.exists(opt.source):
    #     data = 'data/coco2017.yaml'
    #     data_dict = check_dataset(data)  # check if None
    #     opt.source = os.path.join(data_dict['path'],data_dict.get('val_big','val/images'))
    #     assert(os.path.exists(opt.source))

    opt.subsizes = [256,512,768,1024]
    opt.sub_size_scale = 1 #imgsz[1]/images_width==imgsz[0]/images_height in training

    #opt.data = 'data/Guge.yaml'
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    # if not os.path.exists(opt.source):
    #     LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
    #     with torch_distributed_zero_first(LOCAL_RANK):
    #         data_dict = check_dataset(opt.data)  # check if None
    #     source0 = opt.source
    #     source = os.path.join(data_dict.get('path',''), data_dict.get('val_big',''))
    #     print(f'\033[91m{source0} not exist, change to {opt.source}\033[0m')
    # opt.mask_dir = data_dict['mask_dir']
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)



#car_det
    # opt.data = 'data/TZ-car-det-dota.yaml'
    # #opt.weights = 'runs/train/TZ-car_det/exp_TZ-car_det12/weights/best.pt'
    # #opt.weights = 'runs/train/TZ-car_det/exp_TZ-car_det14/weights-1280-512-10epo/best.pt'
    # opt.weights = r'runs/train/TZ-car-det-dota/exp_TZ-car-det-dota4/weights/last.pt'# log
    #opt.weights = r'D:\yolov5\yolov5-rot\runs\train\TZ-car_det\4090-svr-2\weights\epoch39.pt'
    opt.resize = 0
    # #opt.weights = 'runs/train/TZ-car_det/exp_TZ-car_det14/weights-1280-512-5epo-93.65/last.pt'
    # opt.weights = r'D:\yolov5\yolov5-rot\yolov5-rot-sub_size_auto2\runs\TZ-car_det\exp_TZ-car_det16-94.27-67.68\weights\last.pt'
    #mAP50      mAP50:95   f1
    #0.8825     0.6223     0.8753   734.1ms
    # opt.imgsz = [640,640]    #opt.imgsz
    # opt.subsize = [256,256]
    #0.8839     0.6556      0.874   452.1ms
    # opt.imgsz = [1280,1280] #[640,640]
    # opt.subsize = [512,512] #[256,256]
    #0.8604     0.6253     0.8772   571ms
    # opt.imgsz = [1920,1920] #[5,5]
    # opt.subsize = [768,768] #[2,2]
    #0.9116     0.6675     0.8949   908.3ms
    # opt.imgsz = [2560,2560] #[5,5]
    # opt.subsize = [1024,1024] #[2,2]
    #0.8979     0.6697     0.8777   665.0ms
    # opt.imgsz = [1280,2560] #[5,5]
    # opt.subsize = [512,1024] #[2,2]
    #0.8405     0.6184     0.8496   532.6ms
    # opt.imgsz = [640,1280] #[5,5]
    # opt.subsize = [256,512] #[2,2]
    # auto
    #0.8839     0.6556      0.874   455.9ms
    # opt.imgsz = [2560,2560] #[640,640]
    # opt.subsize = [1024,1024] #[256,256]