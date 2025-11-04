# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob
import hashlib
import json
import logging
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.augmentations import augment_hsv, copy_paste, letterbox, mixup, random_perspective, out_range_filt_new, clip_xywhr_rboxes, \
    box_candidates_ioa
from utils.general import check_requirements, check_file, check_dataset, xywh2xyxy, xywhn2xyxy, xyxy2xywhn, \
    xyn2xy, segments2boxes, clean_str, xywhn2xyxy_pts, xyxy2xywhn_pts, xywhr2xyxyxyxy
from utils.torch_utils import torch_distributed_zero_first
from torch.utils.data import Subset # 导入 Subset 类

from utils.plots import Annotator, colors

import math

from image.image_split import CutImages

from tensor.tensor import normalize_dim

from math2.trans import invA23

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    From https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='', shuffle=True,
                      save_dir='',debug_samples=0,sample_count=0,pts=False,data_dict={}):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    if(save_dir==''):
        debug_samples = 0
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix,
                                      save_dir=save_dir,
                                      debug_samples=debug_samples,
                                      pts=pts,
                                      data_dict=data_dict)
    if 'names' in data_dict and data_dict.get('TMax',0):
        dataset.names = data_dict['names']
        nc = data_dict['nc']
        assert len(dataset.names)==nc
        if 'names_vec' in data_dict:
            dataset.names_vec = data_dict['names_vec']
        else:
            root_path = os.path.dirname(os.path.normpath(path if isinstance(path,str) else path[0]))
            names_vec_file = os.path.join(root_path,'names_vec.npz')
            if os.path.exists(names_vec_file):
                dataset.names_vec = torch.load(names_vec_file).to(torch.float32)
                text_embedding_dim = dataset.names_vec.shape[1]
            else:
                if 1:
                    from transformers import CLIPProcessor, CLIPModel
                    # 加载预训练的 CLIP 模型和处理器
                    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    home_path = os.path.expanduser("~")
                    model = CLIPModel.from_pretrained(os.path.join(home_path,"models--openai--clip-vit-base-patch32"))
                    processor = CLIPProcessor.from_pretrained(os.path.join(home_path,"models--openai--clip-vit-base-patch32"))
                    text_embedding_dim = model.config.projection_dim  # 通常是512

                    # 生成词向量
                    inputs = processor(text=dataset.names, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        dataset.names_vec = model.get_text_features(**inputs)  # 形状: [nc, 512]
                else:
                    text_embedding_dim = 512
                    dataset.names_vec = torch.rand((nc,text_embedding_dim),dtype=torch.float32)
                # 归一化（CLIP 相似度计算需归一化）
                dataset.names_vec = normalize_dim(dataset.names_vec,-1) #np.array((len(dataset.names),512))
                torch.save(dataset.names_vec,names_vec_file)  # 'data'是自定义的键名
            assert dataset.names_vec.shape == (len(dataset.names),text_embedding_dim)
    else:
        dataset.names_vec = None

    
    if(sample_count>0):
        if(sample_count < len(dataset)):
            dataset = SubsetRich(dataset, torch.randperm(sample_count))
        else:
            print(f'\033[91m{sample_count} vs {len(dataset)}.\033[0m')

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    nw = 16
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        shuffle = shuffle and sampler is None,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
                        generator = torch.Generator().manual_seed(6148914691236517205)
                        )
    return dataloader, dataset

def create_bigimg_dataloader(image, imgsz, batch_size, subsize=512, over_lap=640,xyoff=[0,0], resize=0, big_warp=1, swaprgb=0):
    dataset = BigImageDataset(image, imgsz, subsize, over_lap, xyoff=xyoff, resize=resize, big_warp=big_warp, swaprgb=swaprgb)
    batch_size = min(batch_size, len(dataset))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.image0_path = Path(self.files[0]).parent.parent / 'images0'
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            # print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            im = cv2.imdecode(np.fromfile(path, dtype=np.uint8),cv2.IMREAD_COLOR)#cv2.imread(path)  # BGR
            assert im is not None, 'Image Not Found ' + path
            # print(f'image {self.count}/{self.nf} {path}: ', end='')

            #im0
            # image0_path = Path(path).parent.parent / 'images0'
            # path = self.img_files[i]
            path0 = self.image0_path / Path(path).name
            # im = cv2.imdecode(np.fromfile(path, dtype=np.uint8),cv2.IMREAD_COLOR)#cv2.imread(path)  # BGR
            im0 = cv2.imdecode(np.fromfile(path0, dtype=np.uint8),cv2.IMREAD_COLOR)#cv2.imread(path)  # BGR
            img0 = np.concatenate([im, im0], axis=-1, dtype=np.uint8) #img0[H,W,6]

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]  #img[H,W,6]

        # Convert
        img = img.transpose((2, 0, 1)) # HWC to CHW, BGR to RGB  #img[H,W,6] -> img[C=6,H,W]
        img[:3,:,:] = img[:3,:,:][::-1,:,:] # HWC to CHW, BGR to RGB
        img[3:,:,:] = img[3:,:,:][::-1,:,:] # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) #img[C=6,H,W]

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

from image.gdal import gdal_start
class LoadBigImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True, subsize=[512,512], over_lap=100, load_label=False):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        ni = len(images)

        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = ni  # number of files
        self.mode = 'image'
        self.auto = auto
        self.subsize = subsize
        self.over_lap = over_lap
        self.load_label = load_label
        assert self.nf > 0, f'No images or videos found in {p}. ' 


    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        if self.subsize[0]>0 and self.subsize[1]>0:
            cut = CutImages(subsize=self.subsize, over_lap=self.over_lap, resize=0)
            img0, cut_imgs = cut.cut_images(path,self.img_size) # BGR
            assert img0 is not None, f'Image Not Found {path}'
        else:
            try:
                img0 = cv2.imread(path)  # BGR
                # img0 = gdal_start(path) #img0[H,W,C]
            except Exception as e:
                img0 = None
            cut_imgs = []
        # cut = CutImages(sub_size=self.sub_size, over_lap=self.over_lap)
        # img0, cut_imgs = cut.cut_images(path) # BGR
        # assert img0 is not None, f'Image Not Found {path}'
        s = f'image {self.count}/{self.nf} {path}: '
        convert_imgs = []
        for sub_item in cut_imgs:
            # Padded resize
            sub_img = sub_item['patch']
            # Convert
            if not cut.resize:
                img = sub_img.transpose((2, 0, 1))  # HWC to CHW, BGR to RGB
            else:
                img = letterbox(sub_img, self.img_size, stride=self.stride, auto=self.auto)[0]
                img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB

            # img = letterbox(sub_img, self.img_size, stride=self.stride, auto=self.auto)[0]
            # # Convert
            # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            convert_imgs.append(img)


        if self.load_label:
            pts_path = self.pts_files[self.count - 1]
            with open(pts_path) as f:
                p = [x.split() for x in f.read().strip().splitlines() if len(x)]
                p = np.array(p, dtype=np.float32)
                p = np.clip(p, 0, 1)
            return path, cut_imgs, convert_imgs, img0, p

        return path, cut_imgs, convert_imgs, img0,  s


    def __len__(self):
        return self.nf  # number of files

class BigImageDataset(Dataset):
    def __init__(self, image, img_size=[640,640], subsize=512, over_lap=100, stride=32, xyoff=[0,0],resize=0, big_warp=1,swaprgb=0):
        cut = CutImages(subsize=subsize, over_lap=over_lap, xyoff=xyoff,resize=resize)
        _, cut_imgs = cut.cut_images(image,img_size) # BGR
        xy = []
        #cuts = []
        #convert_imgs = []
        for sub_item in cut_imgs:
            #sub_img = sub_item['patch']
            #cuts.append(sub_img)
            if not cut.resize:
                xy.append(np.array([sub_item['A23'],sub_item['A23_1'],sub_item['cxy']]))
                #img = sub_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            else:
                xy.append(np.array(sub_item['xy']))
                #img = letterbox(sub_img, img_size, stride=stride, auto=True)[0]
                #img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            #img = np.ascontiguousarray(img)
            #convert_imgs.append(img)
        self.cut = cut
        self.img_size = img_size
        self.xy = xy
        self.image = image
        #self.convert_imgs = convert_imgs
        self.big_warp = big_warp
        self.swaprgb = swaprgb
    
    def __getitem__(self, index):
        xy = self.xy[index]
        if self.cut.resize==0: # 执行仿射变换切图
            A23 = xy[0] #A23[2,3]
            if self.big_warp==0:
                patch = cv2.warpAffine(self.image, A23, (self.img_size[1],self.img_size[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114,114,114))
            else:
                A23_1 = xy[1]
                #image[H,W,3]->patch[self.img_size[1],self.img_size[0],3]
                #cxy1 = A23_1 @ np.array([self.img_size[1]/2,self.img_size[0]/2,1])
                cxy = xy[2,:,2]
                R = max(self.cut.subsize[0],self.cut.subsize[1])
                xylt = np.round(cxy - R).astype(int) #left top coord
                xylt = np.clip(xylt,a_min=0,a_max=None)
                xyrb = xylt + 2*R
                xyrb[0] = np.clip(xylt[0] + 2*R, a_min=None, a_max=self.image.shape[1])
                xyrb[1] = np.clip(xylt[1] + 2*R, a_min=None, a_max=self.image.shape[0])
                patchR = self.image[xylt[1]:xyrb[1], xylt[0]:xyrb[0], :]
                A23t = A23.copy()
                A23t[0,2] += A23t[0,0]*xylt[0] + A23t[0,1]*xylt[1]
                A23t[1,2] += A23t[1,0]*xylt[0] + A23t[1,1]*xylt[1]
                patch = cv2.warpAffine(patchR, A23t, (int(self.img_size[1]),int(self.img_size[0])), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114,114,114))
            
            # patchs[i] = np.ascontiguousarray(patchs[i])
        else:  # 执行resize切图
            x, y = xy[0], xy[1]  # A23[2] -> x, y
            #patchs = image[y:y + subsize[0], x:x + subsize[1]].copy() #patchs[h,w,3]
            patch = self.image[y:y + self.cut.subsize[0], x:x + self.cut.subsize[1]] #patchs[h,w,3]
            # patchs_resize = []
            # for i,xy in enumerate(xys):
            #     patchs_resize.append(letterbox(patchs[i], shape_img, stride=32, auto=True)[0])
            # patchs = np.stack(patchs_resize, axis=0) #patch[B,H,W,3]
            patch = letterbox(patch, self.img_size, stride=32, auto=True)[0]  # patchs[h,w, 3]->ps[img_size[0],img_size[1], 3]
            #patch = np.ascontiguousarray(ps)
        assert patch.shape[:2]==tuple(self.img_size)
        im = patch.transpose((2, 0, 1)) # patch[H,W,3]->im[3,H,W] HWC->CHW  BGR->RGB
        if self.swaprgb:
            im = im[::-1]
        im = np.ascontiguousarray(im)
        
        return xy, im
 
    def __len__(self): 
        return len(self.xy)

class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            if 'youtube.com/' in s or 'youtu.be/' in s:  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.imgs[i] * 0
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
def img2pts_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.pts' for x in img_paths]

def filt_labels_H(labels, least_pixel_size, least_area=40):
    #labels = self.labels[index].copy()#[nobj,1(cls)+4(xyxy))]
    xyxy = labels[:,1:5]#xyxy[nobj,4]
    wh = xyxy[:,2:4]-xyxy[:,0:2]#wh[nobj,2]
    assert wh.shape[0]==0 or torch.any(wh>0), 'must wh>0'
    Lab = torch.norm(wh, p=2, dim=1)
    labels = labels[(Lab > least_pixel_size) & (wh[:,0]*wh[:,1]>=least_area)]
    return labels

def get_bkgrd(bkgrd_big,crop_shape, bkgrd_path=''):
    # 随机选择一个不靠近边缘的点作为旋转中心，避免靠近边缘点
    rows, cols = bkgrd_big.shape[:2]
    if crop_shape[0]>2*rows or crop_shape[1]>2*cols:
        # 边缘的安全距离，避免选择太靠近边缘的点
        RPatch = math.sqrt(crop_shape[0]*crop_shape[0] + crop_shape[1]*crop_shape[1]) / 2
        center_x = random.uniform(RPatch, cols - RPatch) if 2 * RPatch < cols - 1 else cols/2
        center_y = random.uniform(RPatch, rows - RPatch) if 2 * RPatch < rows - 1 else rows/2
        # 使用cv2.getRotationMatrix2D生成仿射矩阵
        #M = cv2.getRotationMatrix2D((center_x, center_y), 0, 1)  # 缩放比例设为1，角度为随机生成
        theta = random.uniform(0,2*math.pi)
        cos_t,sin_t = math.cos(theta),math.sin(theta)
        M = np.array([[cos_t, sin_t, crop_shape[1]/2 - cos_t * center_x - sin_t * center_y],
                    [-sin_t, cos_t, crop_shape[0]/2 + sin_t * center_x - cos_t * center_y]])
        # 生成一个和src相同大小的目标图像dst
        bkgrd = cv2.warpAffine(bkgrd_big, M, (crop_shape[1], crop_shape[0]), borderValue=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        if bkgrd_path!='':
            print(f'[{cols},{rows} , {center_x},{center_y}] {bkgrd_path} RPatch={RPatch}')
            print(M)
            cv2.imshow('Image Display', bkgrd)
            # 等待按键按下，按下任意键关闭窗口
            cv2.waitKey(0)
        return bkgrd
    else:
        return cv2.resize(bkgrd_big,(crop_shape[1],crop_shape[0]))

class SubsetRich(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # 将原始 dataset 的所有属性挂载到子集对象上
        self.__dict__.update(dataset.__dict__)
        self.indices = indices
    # def __len__(self):
    #     return len(self.indices)
class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', save_dir='',debug_samples=0,pts=False,data_dict={},reverse_label=False):
        bkgrd_path=data_dict.get('bkgrd','')
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2] if isinstance(img_size,int) else [-img_size[0] // 2, -img_size[1] // 2]
        self.stride = stride
        self.path = path
        self.pts = pts
        self.data_dict = data_dict
        if 'names' in data_dict:
            self.names = data_dict['names']
        #self.albumentations = Albumentations() if augment else None

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS])
            error = []
            self.image0_path = Path(self.img_files[0]).parent.parent / 'images0'
            for img_ in self.img_files:
                img_ = self.image0_path / Path(img_).name
                if not Path(img_).exists():
                    error.append(img_)
            if len(error) > 0:
                print('\n'.join([str(er) for er in error]))
                raise Exception(f'以上文件找不到配对的差异文件')
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

        # Check cache
        self.suffix = '.cache_v5dfl_diff' if not pts else '.cache_v5dfl_pts_diff'
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(self.suffix)
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == 0.4 and cache['hash'] == get_hash(self.label_files + self.img_files)
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
            if cache['msgs']:
                logging.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int64)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        if(hyp!=None):
            self.least_pixel_size = hyp.get('least_pixel_size', 4) #at least 7 pixels for every object
            self.least_area = hyp.get('least_area', 16)
        else:
            self.least_pixel_size = 4
            self.least_area = 16

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            if isinstance(img_size,int):
                self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int64) * stride
            else:
                # one_shape = np.ceil(np.array(img_size) / stride).astype(np.int64) * stride
                # self.batch_shapes = np.repeat([one_shape],nb,axis=0) #([one_shape].repeat(nb,1)
                self.batch_shapes = np.ceil(np.array(shapes) * max(img_size) / stride + pad).astype(np.int64) * stride
            #self.batch_shapes[nb,2]

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()
        self.save_dir = save_dir
        if isinstance(self.save_dir, Path):
            self.save_dir.mkdir(exist_ok=True, parents=True)
        self.debug_samples = debug_samples

        if self.pts:
            self.xywhn2xyxy = xywhn2xyxy_pts
            self.xyxy2xywhn = xyxy2xywhn_pts
        else:
            self.xywhn2xyxy = xywhn2xyxy
            self.xyxy2xywhn = xyxy2xywhn
        
        #bkgrd
        # 获取没有标签的图像索引列表
        self.neg_ids = [i for i, label in enumerate(self.labels) if label.size == 0]
        if len(self.neg_ids) > 0:
            print(f'\033[92mfind {len(self.neg_ids)} neg images.\033[0m')
        # 第1步：从bkgrd文件夹读取所有图像，并存入cv图像数组bkgrd中
        bkgrd_folder = bkgrd_path if bkgrd_path!='' else str(Path(path if isinstance(path,str) else path[0]).parent) + '/bkgrd' # 图像文件夹路径
        self.bkgrd_images = []  # 用于存储背景图像的列表
        self.bkgrd_paths = []  # 用于存储背景图像的列表
        if os.path.exists(bkgrd_folder):
            # 遍历文件夹中的所有图像文件
            total_files = sum(1 for filename in os.listdir(bkgrd_folder) if filename.endswith('.jpg') or filename.endswith('.png'))
            for filename in tqdm(os.listdir(bkgrd_folder),total=total_files):
                if filename.endswith('.jpg') or filename.endswith('.png'):  # 根据需要添加其他扩展名
                    img_path = os.path.join(bkgrd_folder, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        self.bkgrd_images.append(img)
                        self.bkgrd_paths.append(img_path)
        
        self.reverse_label = reverse_label

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix), repeat(self.pts))),
                        desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if msgs:
            logging.info('\n'.join(msgs))
        if nf == 0:
            logging.info(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = 0.4  # cache version
        try:
            np.save(path, x) #np.save会自动在path后面加上.npy形成保存文件名：labels.cache.npy
            cache_file = path.with_suffix(f'{self.suffix}.npy') #cache_file=labels.cache.npy去掉末尾的.npy后缀还原成path=labels.cache
            if os.path.exists(path): #如果已存在path=labels.cache文件就会报错，所有要先删掉
                path.unlink()
            cache_file.rename(path) #labels.cache.npy->labels.cache 去掉labels.cache.npy末尾的.npy后缀还原成path=labels.cache
            logging.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            logging.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # path not writeable
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self
    def rand_crop_big_bkgrd(self,img_shape):
        bkgrd_id = random.randint(0,len(self.bkgrd_images)-1)
        bkgrd_big = self.bkgrd_images[bkgrd_id]
        return get_bkgrd(bkgrd_big,img_shape,'') #for debug, use self.bkgrd_paths[bkgrd_id]
    def rand_get_a_neg_bkgrd(self,img_shape):
        sel_nid = self.neg_ids[random.randint(0,len(self.neg_ids)-1)]
        path = self.img_files[sel_nid]
        bkgrd = cv2.imdecode(np.fromfile(path, dtype=np.uint8),cv2.IMREAD_COLOR)#cv2.imread(path)  # BGR
        return cv2.resize(bkgrd,(img_shape[1],img_shape[0]))
    def crop_bkgrd(self,img_shape): #img.shape[:2]
        bkgrd_num = len(self.bkgrd_images)
        neg_num = len(self.neg_ids)
        bkgrd = None
        return bkgrd
        if bkgrd_num > 0:
            if neg_num > 0:
                bkgrd = self.rand_crop_big_bkgrd(img_shape) if random.uniform(0, 1)>0.5 else self.rand_get_a_neg_bkgrd(img_shape)
            else:
                bkgrd = self.rand_crop_big_bkgrd(img_shape)
        else:
            if neg_num > 0:
                bkgrd = self.rand_get_a_neg_bkgrd(img_shape)
            else:
                bkgrd = None
        return bkgrd
    
    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            bkgrd = self.crop_bkgrd(self.img_size) #img.shape[:2]
            # Load mosaic
            img, labels,ioa = load_mosaic(self, index, bkgrd)
            #img[h,w,3]  labels[n,5=1+4]
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img2,labels2, ioa2 = load_mosaic(self, random.randint(0, self.n - 1))
                img, labels = mixup(img, labels, img2,labels2)

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            assert img.shape[:2]==tuple(self.img_size)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy() #labels[nt,9=1+4*2]
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = self.xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                bkgrd = self.crop_bkgrd(img.shape[:2])
                img[:,:,:3], labels,ioa,Ar = random_perspective(img[:,:,:3], labels,
                                                 degrees=hyp.get('degrees',0),
                                                 translate=hyp.get('translate',0),
                                                 scale=hyp.get('scale',0.0),
                                                 shear=hyp.get('shear',0),
                                                 perspective=hyp.get('perspective',0),
                                                 flip = [hyp.get('fliplr',0),hyp.get('flipud',0)],
                                                 iou_thr=hyp.get('iou_thr',0.3),
                                                 clip_rate=hyp.get('clip_rate',0.2),
                                                 bkgrd = bkgrd)
                # HSV color-space
                augment_hsv(img, hgain=hyp.get('hsv_h',0.015), sgain=hyp.get('hsv_s',0.7), vgain=hyp.get('hsv_v',0.4))
            else:
                img[:,:,:3], labels,ioa,Ar = random_perspective(img[:,:,:3], labels, #labels[nt,9=1(cls)+4*2]
                                                 degrees=0,
                                                 translate=0,
                                                 scale=0,
                                                 shear=0,
                                                 perspective=0,
                                                 flip = [0, 0],
                                                 iou_thr=0.3,
                                                 clip_rate=0.2)

        im_file = self.img_files[index]
        base_name = os.path.splitext(os.path.basename(im_file))[0]
        a23_path_name = os.path.join(os.path.dirname(os.path.dirname(im_file)),'labels',base_name+'.npy')
        assert os.path.exists(a23_path_name)
        A23 = np.load(a23_path_name)
        A23_r = np.hstack((Ar[:,:2]@A23[:,:2],Ar[:,:2]@A23[:,2:]+Ar[:,2:]))#Ar@A23->A23_r[2,3]
        #reverse_label mode
        if self.reverse_label:#labels[nt,9=1(cls)+4*2]
            pts = labels[:,1:].reshape(-1,2) #labels[nt,9=1+4*2]->pts[nt,8=4*2(xy)]->pts[4nt,2]
            pts_1 = np.hstack([pts, np.ones((pts.shape[0], 1))])  #pts_1[4nt, 3=2+1]
            pts0 = pts_1 @ invA23(A23_r).T  # [4nt, 3] @ [3,2]-->pts0[4nt,2]
            labels[:,1:] = pts0.reshape(-1,8) #pts0[nt,8]
        #
        # after aug show
        if self.debug_samples > 0:
            debug_samples_path = str(self.save_dir)
            f_name = f'{debug_samples_path}/aug[{index}]_{base_name}.jpg'
            if not Path(f_name).exists():
                im0 = img[..., 3:]
                im = img[..., :3]
                if not self.reverse_label:
                    im = np.ascontiguousarray(im) 
                    A_inv = invA23(A23_r)
                    from tools.plotbox import draw_polygon_image_objs
                    draw_polygon_image_objs(im,im0.shape[:2], labels, colors=colors, A23_r = A23_r, thickness=2)
                    # cv2.polylines(im, [pts], isClosed=True, color=(255,255,255), thickness=2)
                    #im0
                    im0 = np.ascontiguousarray(im0) 
                    pts = labels[:,1:].reshape(-1,2) #labels[nt,9=1+4*2]->pts[nt,8=4*2(xy)]->pts[4nt,2]
                    pts_1 = np.hstack([pts, np.ones((pts.shape[0], 1))])  #pts_1[4nt, 3=2+1]
                    pts0 = pts_1 @ A_inv.T  # [4nt, 3] @ [3,2]-->pts0[4nt,2]
                    labels0 = labels.copy()
                    labels0[:,1:] = pts0.reshape(-1,8) #pts0[nt,8]
                    draw_polygon_image_objs(im0,im.shape[:2], labels0, colors=colors, A23_r = A_inv, thickness=2)
                else:
                    ds = Annotator(np.ascontiguousarray(im0), line_width=3, pil=True)            
                    for (cls_, *xyxy) in labels:
                        ds.box_label(xyxy, f'{int(cls_)}', color=colors[int(cls_) % len(colors)])
                    im0 = ds.result()
                im2 = np.concatenate([im0,im], axis=1, dtype=np.uint8)
                cv2.imencode('.jpg', im2)[1].tofile(f_name)
                self.debug_samples-=1

        # if self.label_files[0] is not None:
        #     labels = filt_labels_H(torch.from_numpy(labels),self.least_pixel_size,self.least_area).numpy()
        
        nl = len(labels)  # number of objects
        if nl:
            labels[:, 1:] = self.xyxy2xywhn(labels[:, 1:], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)


        labels_out = torch.zeros((nl, 10 if self.pts else 6))#6/10 = 1(batch index)+1(class)+(4(xywh/4(pts)*2))
        if nl: #shift for batch id in fn_collate
            labels_out[:, 1:] = torch.from_numpy(labels) #labels[nt,9=1(cls)+4*2]->labels_out[10=1(b)+1(cls)+4*2]

        # Convert
        img = img.transpose((2, 0, 1)) #img[H,W,C]-->img[C=6,H,W]
        img[:3,:,:] = img[:3,:,:][::-1,:,:] # HWC to CHW, BGR to RGB
        img[3:,:,:] = img[3:,:,:][::-1,:,:] # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) #img[C,H,W]

        assert ioa.shape[0]==labels_out.shape[0]
        sample = {}
        sample['cls'] = torch.from_numpy(labels[:,0].astype(np.int32)) #cls[nt]
        if hasattr(self, 'names') and self.names_vec is not None:
            sample['name'] = [self.names[int(labels[i, 0].item())] for i in range(labels.shape[0])]
            sample['vec'] = self.names_vec[sample['cls']] #[nt,512]
        sample['A23'] = A23_r
        #
        # 获取文件名，去掉扩展名
        # TMax = self.data_dict.get('TMax',0)
        # if TMax:
        #     assert self.m.tokenizer.eos_token_id==50256
            # 替换 'images' 为 'labels'，并将扩展名从 '.jpg' 改为 '.cmt'
            # cmt_path = os.path.join(os.path.dirname(im_file).replace('images', 'labels'), base_name + '.cmt')
            # if(os.path.exists(cmt_path)):
            #     # 创建一个空列表
            #     lines = []
            #     # 打开文件并逐行读取
            #     with open(cmt_path, 'r') as file:
            #         for line in file:
            #             # 去除换行符并添加到列表中
            #             lines.append(line.strip())
            #     assert nl==len(lines)
            #     sample['name'] = np.zeros((len(lines),TMax),dtype=int) #model_gpt.config.n_embd
            #     # if self.tokenizer is None:
            #     #     self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            #     #     # 这里设置 pad_token 与 eos_token 相同，便于后续处理
            #     #     self.tokenizer.pad_token = self.tokenizer.eos_token
            #     #     self.tokenizer.cls_token = self.tokenizer.eos_token
            #     #     self.tokenizer.mask_token = self.tokenizer.eos_token
            #     #     self.tokenizer.sep_token = self.tokenizer.eos_token
            #     assert self.tokenizer is not None
            #     for i,line in enumerate(lines):
            #         sample['name'][i] = prompt2idx(self.tokenizer,line,TMax).cpu().numpy()#[TMax]
            #         assert np.issubdtype(sample['name'][i].dtype,np.int64)
            #     assert sample['name'].shape[0]==nl
            # else:
            #     assert self.tokenizer.eos_token_id==50256
            #     sample['name'] = np.full((nl,TMax), self.tokenizer.eos_token_id+1)
        #
        return torch.from_numpy(img), labels_out, im_file, shapes, torch.from_numpy(ioa), sample

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, ioa, samples = zip(*batch)  # transposed
        for i, l in enumerate(label):
            assert ioa[i].shape[0]==l.shape[0]
            l[:, 0] = i  # add target image index for build_targets()
        assert len(ioa)==len(label)
        label_batch = torch.cat(label, 0)
        ioa_batch = torch.cat(ioa, 0)
        bnt = label_batch.shape[0]
        assert ioa_batch.shape[0]==bnt
        #
        samples_batch = {}
        keys = samples[0].keys() #keys['im_file','ori_shape','resize_shape','img','cls','bboxes',batch_idx']
        values = list(zip(*[list(b.values()) for b in samples]))#values['im_file',ori_shape,resize_shape,img,cls,bbox,batch_idx]
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb", "vec"}:
                value = torch.cat(value, 0) #value['cls'][bnt]  value['vec'][bnt,512]
            if k in {'name'}:
                # for v in value:
                #     assert np.issubdtype(v.dtype, np.int64)
                value = np.concatenate(value, axis=0) #value[b][nt]->value[bnt]
            if k in {"A23"}:
                # for v in value:
                #     assert np.issubdtype(v.dtype, np.int64)
                value = np.stack(value, axis=0) #value[b][nt]->value[bnt]
            samples_batch[k] = value
        # samples_batch["batch_idx"] = list(samples_batch["batch_idx"])
        # for i in range(len(samples_batch["batch_idx"])):#为每个目标打上batch标记
        #     samples_batch["batch_idx"][i] += i  # add target image index for build_targets()
        # samples_batch["batch_idx"] = torch.cat(samples_batch["batch_idx"], 0)
        if 'name' in samples_batch:#ver
            assert samples_batch['cls'].shape[0]==samples_batch['name'].shape[0]
            for i,(cls,name) in enumerate(zip(samples_batch['cls'],samples_batch['name'])):
                clsid = cls.item()
                assert int(label_batch[i,1].item())==clsid
                # assert self.names_vec[clsid]==name
        #
        assert samples_batch['cls'].shape[0]==bnt
        assert 'name' not in samples_batch or samples_batch['name'].shape[0]==bnt
        return torch.stack(img, 0), label_batch, path, shapes, ioa_batch, samples_batch #samples_batch['vec][bnt,n_embd=512]

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    im = self.imgs[i]
    if im is None:  # not cached in ram
        npy = self.img_npy[i]
        if npy and npy.exists():  # load npy
            im = np.load(npy)
        else:  # read image
            path = self.img_files[i]
            path0 = self.image0_path / Path(path).name
            im = cv2.imdecode(np.fromfile(path, dtype=np.uint8),cv2.IMREAD_COLOR)#cv2.imread(path)  # BGR
            im0 = cv2.imdecode(np.fromfile(path0, dtype=np.uint8),cv2.IMREAD_COLOR)#cv2.imread(path)  # BGR
            im = np.concatenate([im, im0], axis=-1, dtype=np.uint8)
            assert im is not None, 'Image Not Found ' + path
        h0, w0 = im.shape[:2]  # orig hw
        if isinstance(self.img_size, int):#如果是单一长度，则按不变的长宽比缩放
            r = self.img_size / max(h0, w0)  # ratioa
        else:
            r = min(self.img_size[0] / h0, self.img_size[1] / w0)

        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    else:
        return self.imgs[i], self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized


def load_mosaic(self, index, bkgrd=None):
    # loads images in a 4-mosaic
    labels4, segments4, ioa4 = [], [], []
    s = [self.img_size,self.img_size] if(isinstance(self.img_size,int)) else self.img_size
    mosaic_center = self.hyp.get('mosaic_center',0.3)
    yc, xc = s if mosaic_center>1.00001 else [int(random.uniform(-x, 2 * s[i] + x)) for i,x in enumerate(self.mosaic_border)]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    # random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s[0] * 2, s[1] * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s[1] * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s[0] * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s[1] * 2), min(s[0] * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = self.xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            x2a = min(x2a, img4.shape[1])
            y2a = min(y2a, img4.shape[0])
            x1a = max(x1a, 0)
            y1a = max(y1a, 0)
            if self.pts:
                ind,ioa = out_range_filt_new(labels[:, 1:], (x1a, y1a, x2a, y2a), iou_thresh=self.hyp.get('iou_thr',0.3))
                labels,ioa = labels[ind],ioa[ind]
                xy = clip_xywhr_rboxes(labels[:, 1:].copy(), x1a, y1a, x2a, y2a, clip_rate=self.hyp.get('clip_rate',0.2))
                labels[:, 1:] = xy
            else:
                ind = box_candidates_ioa(boxes=labels[:, 1:], width=x2a, height=y2a, iou_thr=self.hyp.get('iou_thr',0.3), 
                                         w_min=x1a, h_min=y1a)
                labels = labels[ind]
                xyxy = labels[:, 1:].copy()
                xyxy[:, 0] = np.clip(xyxy[:, 0], x1a, x2a)  # x1
                xyxy[:, 1] = np.clip(xyxy[:, 1], y1a, y2a)  # y1
                xyxy[:, 2] = np.clip(xyxy[:, 2], x1a, x2a)  # x2
                xyxy[:, 3] = np.clip(xyxy[:, 3], y1a, y2a)  # y2
                labels[:, 1:] = xyxy
                ioa = np.ones((labels.shape[0]),dtype=float)
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        else:
            ioa = np.ones((labels.shape[0]),dtype=float)
        labels4.append(labels)
        ioa4.append(ioa)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    ioa4 = np.concatenate(ioa4, 0)
    assert ioa4.shape[0]==labels4.shape[0]

    if 0:
        name = os.path.splitext(os.path.basename(self.img_files[indices[0]]))[0]
        debug_samples_path = str(self.save_dir)
        f_name = f'{debug_samples_path}/aug[{indices[0]}]_{name}_whole.jpg'
        if self.debug_samples > 0:
            if not Path(f_name).exists():
                ds = Annotator(img4, line_width=3, pil=True)            
                for (cls_, *xyxy) in labels4:
                    ds.box_label(xyxy, f'{int(cls_)}', color=colors[int(cls_) % len(colors)])
                im0 = ds.result()            
                cv2.imencode('.jpg', im0)[1].tofile(f_name)
                # self.debug_samples-=1

    if self.pts:
        if labels4.shape[0] > 0:
            ind,ioa = out_range_filt_new(labels4[:, 1:], (img4.shape[1], img4.shape[0]), self.hyp.get('iou_thr',0.3),ioa4)
            labels4,ioa = labels4[ind],ioa[ind]
        else:
            ioa = np.ones((0),dtype=float)
        assert ioa.shape[0]==labels4.shape[0]
        if len(segments4) > 0:
            segments4 = segments4[i]
    else:
        for x in (*labels4[:, 1:], *segments4):
            # np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
            np.clip(x[0::2], 0, 2 * s[1], out=x[0::2])  # clip when using random_perspective()
            np.clip(x[1::2], 0, 2 * s[0], out=x[1::2])  # clip when using random_perspective()
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])

    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment img4[C,2H,2W]->img4[C,H,W]
    img4, labels4,ioa,Ar = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp.get('degrees',0),
                                       translate = mosaic_center*self.hyp.get('translate',0),
                                       scale=self.hyp.get('scale',0.0),
                                       shear=self.hyp.get('shear',0),
                                       perspective=self.hyp.get('perspective',0),
                                       flip = [self.hyp.get('fliplr',0),self.hyp.get('flipud',0)],
                                       border=self.mosaic_border, # border to remove
                                       iou_thr=self.hyp.get('iou_thr',0.3),
                                       clip_rate=self.hyp.get('clip_rate',0.2),
                                       bkgrd = bkgrd)
    # HSV color-space
    augment_hsv(img4, hgain=self.hyp.get('hsv_h',0.015), sgain=self.hyp.get('hsv_s',0.7), vgain=self.hyp.get('hsv_v',0.4))

    assert labels4.shape[0]==ioa.shape[0]
    return img4, labels4, ioa


def load_mosaic9(self, index):
    # loads images in a 9-mosaic

    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (*labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9, ioa,Ar = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp.get('degrees',0),
                                       translate=self.hyp.get('translate',0),
                                       scale=self.hyp.get('scale',0.0),
                                       shear=self.hyp.get('shear',0),
                                       perspective=self.hyp.get('perspective',0),
                                       flip = [self.hyp.get('fliplr',0),self.hyp.get('flipud',0)],
                                       border=self.mosaic_border,
                                       iou_thr=self.hyp.get('iou_thr',0.3))  # border to remove
    # HSV color-space
    augment_hsv(img, hgain=self.hyp.get('hsv_h',0.015), sgain=self.hyp.get('hsv_s',0.7), vgain=self.hyp.get('hsv_v',0.4))

    return img9, labels9


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../datasets/coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../datasets/coco128'):  # from utils.datasets import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../datasets/coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sum([list(path.rglob(f"*.{img_ext}")) for img_ext in IMG_FORMATS], [])  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # add image to txt file


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix, pts = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    Image.open(im_file).save(im_file, format='JPEG', subsampling=0, quality=100)  # re-save image
                    msg = f'{prefix}WARNING: corrupt JPEG restored and saved {im_file}'

        # verify labels
                # verify labels        
        lb_file = Path(lb_file)
        pol_file = lb_file.with_suffix('.pol')
        if os.path.exists(pol_file):
            nf = 1  # label found
            with open(pol_file) as f: #.pts
                pol_p = [np.array(x.split(), dtype=np.float32) for x in f.read().strip().splitlines() if len(x.strip())]    # cls x1,y1,x2,y2...,xn,yn
            nl = len(pol_p)
            p = []
            l = []
            if nl:
                for pol in pol_p:
                    x1, y1, x2, y2 = min(pol[1::2]), min(pol[2::2]), max(pol[1::2]), max(pol[2::2])
                    l.append([pol[0], (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
                    p.append(pol[1:].reshape(-1, 2))
                l = np.array(l, dtype=np.float32)
                # 排好顺逆统一
                rboxes = []
                for pts_ in p:
                    # NOTE: Use cv2.minAreaRect to get accurate xywhr,
                    # especially some objects are cut off by augmentations in dataloader.
                    (cx, cy), (w, h), angle = cv2.minAreaRect(pts_)
                    rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
                p = xywhr2xyxyxyxy(np.array(rboxes, dtype=np.float32)).reshape(-1, 8)
                assert(p.shape[0]==l.shape[0])#pts的行数即目标数  和  lables里的行数目标数应该一致

                clss = l[:, 0].reshape(-1, 1)
                l = np.concatenate((clss, p), axis=1)#把pts数据p拼到原始yolo标签l后面:l[nt,5] cat p[nt,4*2=8]-->p[nt,5 + 4*2=8]
                l = np.unique(l, axis=0)  # remove duplicate rows,  同一目标重复标注，合并
            else:
                nm = 1  # label missing
                l = np.zeros((0, 9 if pts else 5), dtype=np.float32)
        elif os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, 'r') as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any([len(x) > 8 for x in l]):  # is segment
                    classes = np.array([x[0] for x in l], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                    l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                l = np.array(l, dtype=np.float32)
            nl = len(l)
            if nl:
                assert l.shape[1] == 5, 'labels require 5 columns each'
                # assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                # assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                            #add... 读取新增pts文件中的数据
                if pts:
                    pts_file = Path(lb_file).with_suffix('.pts')
                    if pts_file.exists():
                        with open(pts_file) as f: #.pts
                            p = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    else:
                        p = [['-']] * len(l)
                        #p = [['0','0','0','0','0','0','0','0'] for x in p if x==['-']]]
                    for i in range(len(p)):
                        if p[i]==['-']:#无数据，针对非旋转目标的情况
                            xc,yc,w,h = float(l[i][1]),float(l[i][2]),float(l[i][3]),float(l[i][4])
                            #p[i] = [str(round(xc-w/2,6)),'0','0','0','0','0','0','0']
                            p[i] = [str(round(xc-w/2,6)),str(round(yc-h/2,6)),
                                    str(round(xc+w/2,6)),str(round(yc-h/2,6)),
                                    str(round(xc+w/2,6)),str(round(yc+h/2,6)),
                                    str(round(xc-w/2,6)),str(round(yc+h/2,6))]
                            #由于后面p = np.array(p, dtype=np.float32)要求字符串必须是数值的，'-'转数据矩阵会报错
                            #也就是水平框的四个顶点构建四边形
                    p = np.array(p, dtype=np.float32)#list转np矩阵[nt,4*2=8]
                    # 排好顺逆统一
                    rboxes = []
                    p = p.reshape(len(p), -1, 2)
                    for pts in p:
                        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
                        # especially some objects are cut off by augmentations in dataloader.
                        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
                        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
                    p = xywhr2xyxyxyxy(np.array(rboxes, dtype=np.float32)).reshape(-1, 8)
                    assert(p.shape[0]==l.shape[0])#pts的行数即目标数  和  lables里的行数目标数应该一致

                    clss = l[:, 0].reshape(-1, 1)
                    l = np.concatenate((clss, p), axis=1)#把pts数据p拼到原始yolo标签l后面:l[nt,5] cat p[nt,4*2=8]-->p[nt,5 + 4*2=8]
                    l = np.unique(l, axis=0)  # remove duplicate rows,  同一目标重复标注，合并
                    assert (l[:, 0] >= 0).all(), 'negative cls labels'
                    if len(l) < nl:
                        segments = np.unique(segments, axis=0)
                        msg = f'{prefix}WARNING: {im_file}: {nl - len(l)} duplicate labels removed'
                else:
                    assert (l >= 0).all(), 'negative labels'
            else:
                ne = 1  # label empty
                l = np.zeros((0, 9 if pts else 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 9 if pts else 5), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """ Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.datasets import *; dataset_stats('../datasets/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *[round(x, 4) for x in points]] for c, *points in labels]

    def unzip(path):
        # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
        if str(path).endswith('.zip'):  # path is data.zip
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            assert os.system(f'unzip -q {path} -d {path.parent}') == 0, f'Error unzipping {path}'
            dir = path.with_suffix('')  # dataset directory
            return True, str(dir), next(dir.rglob('*.yaml'))  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    def hub_ops(f, max_dim=1920):
        # HUB ops for 1 image 'f'
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)  # ratio
        if r < 1.0:  # image too large
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(im_dir / Path(f).name, quality=75)  # save

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_file(yaml_path), errors='ignore') as f:
        data = yaml.safe_load(f)  # data dict
        if zipped:
            data['path'] = data_dir  # TODO: should this be dir.resolve()?
    check_dataset(data, autodownload)  # download dataset if missing
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}  # statistics dictionary
    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # load dataset
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        x = np.array(x)  # shape(128x80)
        stats[split] = {'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()},
                        'image_stats': {'total': dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()),
                                        'per_class': (x > 0).sum(0).tolist()},
                        'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in
                                   zip(dataset.img_files, dataset.labels)]}

        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files), total=dataset.n, desc='HUB Ops'):
                pass

    # Profile
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)  # save stats *.json
            t2 = time.time()
            with open(file, 'r') as f:
                x = json.load(f)  # load hyps dict
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

    # Save, print and return
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)  # save stats.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats


def resize_and_save_images(data_path, num=64, imgsz=[640,640]):
    src_folder = data_path + '/images'
    dst_folder = data_path + '/qnt_imgs'
    # 确保目标文件夹存在
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 支持的图像扩展名
    extensions = ['.jpg', '.png', '.bmp', '.tif']
    
    # 获取所有符合条件的图像文件
    image_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder)
                   if os.path.splitext(f)[1].lower() in extensions]
    
    # 随机选择指定数量的图像文件
    selected_images = random.sample(image_files, num)
    
    # 调整图像大小并保存到目标文件夹
    for i, image_file in tqdm(enumerate(selected_images)):
        with Image.open(image_file) as img:
            try:
                resized_img = img.resize((imgsz[1], imgsz[0]), Image.Resampling.LANCZOS)
            except AttributeError:
                # 如果失败，回退到旧的 Image.ANTIALIAS
                resized_img = img.resize((imgsz[1], imgsz[0]), Image.ANTIALIAS)
            # 保存图像到目标文件夹
            base_name = os.path.basename(image_file)
            resized_img.save(os.path.join(dst_folder, base_name))
            #print(f"Saved {i+1}/{num}: {os.path.join(dst_folder, base_name)}")


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
def show_ft(img, xy_rect=(),ft_labels=(),cls=()):
    # h, w, _ = img.shape
    assert len(cls)==0 or len(cls)==xy_rect.shape[0]
    new_img = img.copy()
    theta_fine = np.linspace(0, 2*np.pi, 200)
    for i, xy_label in enumerate(xy_rect):
        c = int(cls[i])
        color = colors[c % len(colors)] if len(cls)>0 else (0,0,255)
        if len(ft_labels) > 0:#for ft
            label = ft_labels[c]
            an,bn,cn,dn = [abcd.reshape(-1) for abcd in np.split(label[2:].reshape(-1, 4), 4, axis=-1)]
            x_approx = sum([an[i]*np.cos((i+1)*theta_fine) + bn[i]*np.sin((i+1)*theta_fine) for i in range(an.shape[0])])
            y_approx = sum([cn[i]*np.cos((i+1)*theta_fine) + dn[i]*np.sin((i+1)*theta_fine) for i in range(an.shape[0])])
            xy = np.vstack([x_approx + label[0], y_approx + label[1]]).T# * (w, h)
            xy = xy.astype(np.int32)
            cv2.polylines(new_img, [xy], True, color, 2)
        if xy_label.shape[-1] == 4:#for h rect[x,y,w,h]
            cv2.rectangle(new_img, (int(xy_label[0]), int(xy_label[1])), (int(xy_label[2]), int(xy_label[3])), (0,255,0), 2)
        elif xy_label.shape[-1] == 8:#for h pts 4(x,y,w,h)+4*2
            pts_ = xy_label.reshape([-1, 2]).astype(np.int32)
            cv2.polylines(new_img, [pts_], True, color, 2)
        elif xy_label.shape[-1] == 12:#for h pts 4(x,y,w,h)+4*2
            pts_ = xy_label[4:].reshape([-1, 2]).astype(np.int32)
            cv2.polylines(new_img, [pts_], True, color, 2)
            #
            # 计算四边形的中心点
            center = np.mean(pts_, axis=0).astype(np.int32)
            # 计算前两个点的中点
            mid_point = np.mean(pts_[:2], axis=0).astype(np.int32)
            # 绘制箭头
            cv2.arrowedLine(new_img, tuple(center), tuple(mid_point), (0, 255, 0), 1, tipLength=0.2)
  
    return new_img