# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
General utils
"""

import contextlib
import glob
import logging
import math
import os
import platform
import random
import re
import signal
import time
import urllib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml
import sys

from utils.downloads import gsutil_getsize
from utils.metrics import box_iou, fitness, batch_probiou
from utils.torch_utils import init_torch_seeds

from tensor.tensor import normalize_dim

from general.global_cfg import replace_path

# from DOTA_devkit.polyiou_cpu import poly_nms_cpu64

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory

class Profile(contextlib.ContextDecorator):
    # Usage: @Profile() decorator or 'with Profile():' context manager
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        print(f'Profile results: {time.time() - self.start:.5f}s')


class Timeout(contextlib.ContextDecorator):
    # Usage: @Timeout(seconds) decorator or 'with Timeout(seconds):' context manager
    def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._timeout_handler)  # Set handler for SIGALRM
        signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # Cancel SIGALRM if it's scheduled
        if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
            return True


def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler


def methods(instance):
    # Get class/instance methods
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]


def set_logging(rank=-1, verbose=True):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if (verbose and rank in [-1, 0]) else logging.WARN)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def is_docker():
    # Is environment a Docker container?
    return Path('/workspace').exists()  # or Path('/.dockerenv').exists()


def is_colab():
    # Is environment a Google Colab instance?
    try:
        import google.colab
        return True
    except Exception as e:
        return False


def is_pip():
    # Is file in a pip package?
    return 'site-packages' in Path(__file__).absolute().parts


def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters?
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def file_size(file):
    # Return file size in MB
    return Path(file).stat().st_size / 1e6


def check_online():
    # Check internet connectivity
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False


@try_except
def check_git_status():
    # Recommend 'git pull' if code is out of date
    msg = ', for updates see https://github.com/ultralytics/yolov5'
    print(colorstr('github: '), end='')
    assert Path('.git').exists(), 'skipping check (not a git repository)' + msg
    assert not is_docker(), 'skipping check (Docker image)' + msg
    assert check_online(), 'skipping check (offline)' + msg

    cmd = 'git fetch && git config --get remote.origin.url'
    url = check_output(cmd, shell=True, timeout=5).decode().strip().rstrip('.git')  # git fetch
    branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # checked out
    n = int(check_output(f'git rev-list {branch}..origin/master --count', shell=True))  # commits behind
    if n > 0:
        s = f"⚠️ YOLOv5 is out of date by {n} commit{'s' * (n > 1)}. Use `git pull` or `git clone {url}` to update."
    else:
        s = f'up to date with {url} ✅'
    print(emojis(s))  # emoji-safe


def check_python(minimum='3.6.2'):
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name='Python ')


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    if hard:  # assert min requirements met
        assert result, f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'
    else:
        return result
    

TORCH_1_13 = check_version(torch.__version__, "1.13.0")
TORCH_2_4 = check_version(torch.__version__, "2.4.0")
TORCH_1_9 = check_version(torch.__version__, "1.9.0")

@try_except
def check_requirements(requirements='requirements.txt', exclude=(), install=True):
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  # check python version
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
        requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:  # DistributionNotFound or VersionConflict if requirements not met
            s = f"{prefix} {r} not found and is required by YOLOv5"
            if install:
                print(f"{s}, attempting auto-update...")
                try:
                    assert check_online(), f"'pip install {r}' skipped (offline)"
                    print(check_output(f"pip install '{r}'", shell=True).decode())
                    n += 1
                except Exception as e:
                    print(f'{prefix} {e}')
            else:
                print(f'{s}. Please install and rerun your command.')

    if n:  # if packages updated
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


def check_imshow():
    # Check if environment supports image displays
    try:
        assert not is_docker(), 'cv2.imshow() is disabled in Docker environments'
        assert not is_colab(), 'cv2.imshow() is disabled in Google Colab environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


def check_file(file):
    # Search/download file (if necessary) and return path
    file = str(file)  # convert to str()
    if Path(file).is_file() or file == '':  # exists
        return file
    elif file.startswith(('http:/', 'https:/')):  # download
        url = str(Path(file)).replace(':/', '://')  # Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file)).name.split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, file)
        assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file
    else:  # search
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def check_dataset(data, autodownload=True):
    # Download and/or unzip dataset if not found locally
    # Usage: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip

    # Download (optional)
    extract_dir = ''
    if isinstance(data, (str, Path)) and str(data).endswith('.zip'):  # i.e. gs://bucket/dir/coco128.zip
        download(data, dir='../datasets', unzip=True, delete=False, curl=False, threads=1)
        data = next((Path('../datasets') / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)  # dictionary

    # Parse yaml
    data["path"] = replace_path(data["path"])
    path = extract_dir or Path(data.get('path') or '')  # optional 'path' default to '.'
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data["path"] = path
    if(not os.path.exists(path)):
        print(f'\033[91m{path} not exists.\033[0m')
        sys.exit()
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            #data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]
            if isinstance(data[k], str):
                data[k] = str(path / data[k])
                # 将路径拆分成目录和文件名部分
                data_path, basename = os.path.split(data[k])
                # 将"images"替换为"labels"
                lables_path = os.path.join(data_path, "labels")
                if not os.path.exists(lables_path):
                    print(f'\033[91m{k} path:{lables_path} not exists.\033[0m')
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]
                for x in data[k]:
                    if not os.path.exists(x):
                        print(f'\033[91m{k} path:{x} not exists.\033[0m')

    # assert 'nc' in data, "Dataset 'nc' key missing."
    # data['nc'] = len(data['names'])
    # if 'names' not in data:
    #     data['names'] = [f'class{i}' for i in range(data['nc'])]  # assign class names if missing
    if 'names'in data and len(data['names'])>0:
        data['nc'] = len(data['names'])
    else:
        train_images = data['train'] if isinstance(data['train'],str) else data['train'][0]
        names_path = os.path.join(os.path.dirname(train_images.rstrip('/\\')),'names.txt')
        if os.path.exists(names_path):
            with open(names_path, "r") as f:
                data['names'] = [line.strip() for line in f if line.strip()]
            data['nc'] = len(data['names'])
        else:
            if 'nc' in data:
                data['names'] = [f'class{i}' for i in range(data['nc'])]  # assign class names if missing
            else:
                print(f'\033[91mnc not in data.\033[0m')
    train, val, test, s = [data.get(x) for x in ('train', 'val', 'test', 'download')]
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and autodownload:  # download script
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # filename
                    print(f'Downloading {s} ...')
                    torch.hub.download_url_to_file(s, f)
                    root = path.parent if 'path' in data else '..'  # unzip directory i.e. '../'
                    Path(root).mkdir(parents=True, exist_ok=True)  # create root
                    r = os.system(f'unzip -q {f} -d {root} && rm {f}')  # unzip
                elif s.startswith('bash '):  # bash script
                    print(f'Running {s} ...')
                    r = os.system(s)
                else:  # python script
                    r = exec(s, {'yaml': data})  # return None
                print('Dataset autodownload %s\n' % ('success' if r in (0, None) else 'failure'))  # print result
            else:
                raise Exception('Dataset not found.')

    return data  # dictionary

def get_source(source,data):
    if os.path.exists(source):
        return source
    else:
        data_dict = check_dataset(data)  # check if None
        assert(os.path.exists(data_dict['val']))
        if not os.path.isdir(data_dict['val']):
            data_dict['detect'] = os.path.join(os.path.dirname(data_dict['val']),data_dict.get('detect','images'))
        else:
            data_dict['detect'] = data_dict['val']
        assert(os.path.isdir(data_dict['detect']))
        return data_dict['detect']

def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):
    # Multi-threaded file download and unzip function, used in data.yaml for autodownload
    def download_one(url, dir):
        # Download 1 file
        f = dir / Path(url).name  # filename
        if Path(url).is_file():  # exists in current path
            Path(url).rename(f)  # move to dir
        elif not f.exists():
            print(f'Downloading {url} to {f}...')
            if curl:
                os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")  # curl download, retry and resume on fail
            else:
                torch.hub.download_url_to_file(url, f, progress=True)  # torch download
        if unzip and f.suffix in ('.zip', '.gz'):
            print(f'Unzipping {f}...')
            if f.suffix == '.zip':
                s = f'unzip -qo {f} -d {dir}'  # unzip -quiet -overwrite
            elif f.suffix == '.gz':
                s = f'tar xfz {f} --directory {f.parent}'  # unzip
            if delete:  # delete zip file after unzip
                s += f' && rm {f}'
            os.system(s)

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multi-threaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0) # labels[n][nt,5]->labels[n*nt,5] [class xywh]
    classes = labels[:, 0].astype(np.int64)  #labels[n*nt,5]->classes[n*nt]
    weights = np.bincount(classes, minlength=nc) #计数统计出每类出现的次数classes[n*nt]->weights[nc]

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1000000  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80),master_mask=None,slave_rate=1.0):
    # Produces image weights based on class_weights and image contents
    #labels[nimages][nt,1(cls)+4(box)+4(pts)*2]
    #class_counts = np.array([np.bincount(x[:, 0].astype(np.int32), minlength=nc) for x in labels])
    # 计算每个图像的类别计数
    class_counts = np.array([
        np.bincount(
            np.where((x[:, 0] >= 0) & (x[:, 0] < nc), x[:, 0].astype(np.int32), 0),  # 超出范围的值设为 -1
            minlength=nc
        ) for x in labels
    ])
    #class_counts[nimages,nc]
    # 初始化权重矩阵
    if master_mask is not None:
        weights = np.zeros_like(class_counts, dtype=np.float32)
        assert len(master_mask) == len(weights), "master_mask 和 weights 的长度不匹配"
        weights[master_mask] = (class_weights.reshape(1, nc) * class_counts[master_mask])
        #
        image_weights = np.zeros(len(labels), dtype=np.float32)
        image_weights[master_mask] = weights[master_mask].sum(1)
        #
        master_sum = image_weights[master_mask].sum()
        slave_sum = slave_rate * master_sum
        master_count = master_mask.sum()
        slave_count = len(master_mask) - master_count
        # slave_rate = slave_count/master_count
        #weights[~master_mask] = slave_sum / slave_count
        #average_class_weight = class_weights.mean() #np.median(class_weights)#
        #weights[~master_mask] = slave_rate * average_class_weight * class_counts[~master_mask]
        #class_weights[nc]和class_counts[nimages,nc]的每一行做乘法，然后按行求和.sum(1)-->image_weights[nimages]
        objn = class_counts[~master_mask].sum(1) #objn[nimg]
        assert objn.shape[0]==slave_count
        obj_total = objn.sum()
        image_weights[~master_mask] = slave_sum * objn / obj_total
        #image_weights = weights.sum(1)
        #
        if 0:
            pos_w = image_weights[master_mask].sum()
            neg_w = image_weights[~master_mask].sum()
    else:
        image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    #image_weights[nimages]
    # index = random.choices(range(nimages), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


# def xywh2xyxy(x):
#     # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
#     y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
#     y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
#     y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
#     return y

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    return (np.concatenate if isinstance(x, np.ndarray) else torch.cat)((xy - wh, xy + wh), -1)


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

def xywhn2xyxy_pts(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0::2] = w * x[:, 0::2] + padw
    y[:, 1::2] = h * x[:, 1::2] + padh
    return y

def xyxy2xywhn_pts(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    # if clip:
    #     clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0::2] = x[:, 0::2] / w
    y[:, 1::2] = x[:, 1::2] / h
    return y

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0::2] -= pad[0]  # x padding
    coords[:, 1::2] -= pad[1]  # y padding
    coords /= gain
    return coords
def scale_coords_poly(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, ::2] -= pad[0]  # x padding
    coords[:, 1::2] -= pad[1]  # y padding
    coords[:, :8] /= gain
    return coords
def apply_affine_transform(det, A23):
    assert det.shape[-1]==6 or det.shape[-1]==10
    pts_size = det.shape[-1] - 2 #[1(conf)+1(cls)]
    # 确保A23是一个torch.float32类型的2x3矩阵
    #A23 = torch.tensor(A23, dtype=torch.float32).reshape(2, 3)
    
    # 提取仿射变换矩阵的旋转和位移部分
    R = A23[:, :2]
    t = A23[:, 2]
    
    num_targets = det.shape[0]
    
    for i in range(num_targets):
        points = det[i, :pts_size].reshape(-1, 2)  # 提取4个点 (x, y)
        
        # 使用矩阵乘法和位移进行仿射变换
        transformed_points = torch.matmul(points, R.T) + t
        
        # 更新原始张量中的点
        det[i, :pts_size] = transformed_points.flatten()
    
    return det

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def non_max_suppression_dfl(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                            labels=(), max_det=300, return_indices=False):
    # Runs Non-Maximum Suppression (NMS) on inference results
    # prediction: [batch, ntotal, 5+nc]
    # Returns:
    #      list of detections, on (n,6) tensor per image [xyxy, conf, cls]  
    # output indices  
    indices_grid = torch.arange(0,prediction.shape[1], device=prediction.device).long() #indices_grid[ntotal]

    nc = prediction.shape[2] - 4  # number of classes
    if isinstance(conf_thres, float) or len(conf_thres.shape)==0:
        conf_thres = torch.ones(nc, device=prediction.device) * conf_thres
    else:
        conf_thres = conf_thres.to(prediction.device)
    xc = prediction[..., 4:].max(-1)[0] > conf_thres.min() # candidates xc[b,ntotal]

    # Checks
    assert ((0 <= conf_thres) & (conf_thres <= 1)).all(), f'Invalid Confidence threshold [{conf_thres.min()}, {conf_thres.max()}], valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output_indices = [torch.zeros((0, 1), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        #prediction[b,ntotal,4+nc]->x[ntotal,4+nc]
        #xc[b,ntotal] xc[xi][ntotal]->x[nt,4+nc]
        # assert xc[xi].shape[0] == x.shape[0]
        # assert xc[xi].shape[0] == indices_grid.shape[0]
        x = x[xc[xi]]  # x[ntotal,4+nc]->x[nt,4+nc]
        grid = indices_grid[xc[xi]] #indices_grid[ntotal]->grid[nt]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        # x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # x[np,4(xywh)+1(obj)+cls]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4]) #box[nt,4]
        # x[:,:4(xywh)] -> box[np,4(xyxy)]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:#
            i, j = (x[:, 4:] > conf_thres[None]).nonzero(as_tuple=False).T #i[np]目标行号, j[np]类列号
            x = torch.cat((box[i], x[i, j + 4, None], j[:, None].float()), 1)
            grid = grid[i]
        else:  # best class only
            conf, j = x[:, 4:].max(1, keepdim=True) #x[np,4+nc]->conf[np,1],j[np,1]
            filter_mask = (conf > conf_thres[j]).view(-1) #filter_mask[np]
            x = torch.cat((box, conf, j.float()), 1) #box[np,4(xyxy)],conf[np,1],j[np,1]->x[np,6=4(xyxy)+1(conf)+1(cls)]
            # assert filter_mask.shape[0] == x.shape[0]
            # assert filter_mask.shape[0] == grid.shape[0]
            x = x[filter_mask] #x[np,6=4(xyxy)+1(conf)+1(cls)]->x[np2,6=4(xyxy)+1(conf)+1(cls)]
            grid = grid[filter_mask] #grid[np]->grid[np2]

        # Filter by class
        if classes is not None:
            grid = grid[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            grid = grid[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes[nt,4] (offset by class), scores[nt]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # i[nt_nms]
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]        
        # assert i.max() < boxes.shape[0]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights [i, n] [1, n]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output_indices[xi] = grid[i]
        output[xi] = x[i] #x[nt,6]->output[xi][nt_nms,6]
        # if (time.time() - t) > time_limit:
        #     print(f'WARNING: NMS time limit {time_limit}s exceeded')
        #     break  # time limit exceeded
    return output#output[b][nt,6=4(xyxy)+1(conf)+1(cls)], output_indices if return_indices else output

def non_max_suppression_txt(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                            labels=(), max_det=300, model=None, names_vec=None):
    # Runs Non-Maximum Suppression (NMS) on inference results
    # prediction: [b, ntotal, 4(box)+nc + TMax*n_embd]
    # Returns:
    #      list of detections, on (n,4(box)+512(n_embd)+1(conf)+1(cls)) tensor per image [xyxy, conf, cls]  
    # output indices  
    indices_grid = torch.arange(0,prediction.shape[1], device=prediction.device).long() #indices_grid[ntotal]

    if names_vec is not None:
        TMax, n_embd = 1, 512 #for CLIP
    else:
        TMax, n_embd = 0, 0
    nc = prediction.shape[2] - TMax*n_embd - 4  # number of classes
    assert names_vec==None or (nc==1 and names_vec.shape[-1]==n_embd) or names_vec.shape==(nc,n_embd)

    if isinstance(conf_thres, float) or len(conf_thres.shape)==0:
        conf_thres = torch.ones(nc, device=prediction.device) * conf_thres
    else:
        if isinstance(conf_thres,np.ndarray):
            conf_thres = torch.from_numpy(conf_thres)
        conf_thres = conf_thres.to(prediction.device)
    nc_matrix = prediction[..., 4:-TMax*n_embd] if names_vec is not None else prediction[..., 4:] #nc_matrix[b,ntotal,nc]
    assert nc_matrix.shape[-1]==nc
    xc = nc_matrix.max(-1)[0] > conf_thres.min() # candidates nc_matrix[b,ntotal,nc] -> xc[b,ntotal]
    assert nc == prediction.shape[-1]-4-TMax*n_embd

    # Checks
    assert ((0 <= conf_thres) & (conf_thres <= 1)).all(), f'Invalid Confidence threshold [{conf_thres.min()}, {conf_thres.max()}], valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    max_wh = 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    m = model.get_module_byname('YoloText') if model is not None else None
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output_indices = [torch.zeros((0, 1), device=prediction.device)] * prediction.shape[0]
    
    for xi, x_text in enumerate(prediction):  # image index, image inference
        #prediction[b,ntotal,4+nc]->x_text[ntotal,4+nc + TMax*n_embd]
        #xc[b,ntotal] xc[xi][ntotal]->x_text[nt,4+nc + TMax*n_embd]
        # assert xc[xi].shape[0] == x_text.shape[0]
        # assert xc[xi].shape[0] == indices_grid.shape[0]
        x_text = x_text[xc[xi]]  # x_text[np,4+nc + TMax*n_embd]->x_text[np,4+nc + TMax*n_embd]
        grid = indices_grid[xc[xi]] #indices_grid[ntotal]->grid[nt]
        #
        if TMax>0:
            x = x_text[...,:-TMax*n_embd] #x[np,4+nc]
            ptext = x_text[...,-TMax*n_embd:] #ptext[np,TMax*n_embd]
        else:
            x = x_text#x[np,5(xywhr)+nc]
            ptext = None

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4]) ##x[np,533=4(xywh)+nc+nembd(512)]->box[np,4(xyxy)]
        # x[:,:4(xywh)] -> box[np,4(xyxy)]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:#
            i, j = (x[:, 4:] > conf_thres[None]).nonzero(as_tuple=False).T #i[np]目标行号, j[np]类列号
            if ptext is not None:
                x = torch.cat((box[i], ptext[i], x[i, j + 4, None], j[:, None].float()), 1) #-> x[npi, 518 = 4(box)+512(n_embd) + 1(conf) + 1(cls)]
            else:
                x = torch.cat((box[i], x[i, j + 4, None], j[:, None].float()), 1) #-> x[npi, 6 = 4(box) + 1(conf) + 1(cls)]
            grid = grid[i]
        else:  # best class only
            conf, j = x[:, 4:].max(1, keepdim=True) #x[np,4+nc]->conf[np,1],j[np,1]
            filter_mask = (conf > conf_thres[j]).view(-1) #filter_mask[np]
            x = torch.cat((box, ptext, conf, j.float()), 1) #box[np,4(xyxy)],conf[np,1],j[np,1]->x[np,518 = 4(xyxy)+512(n_embd)+1(conf)+1(cls)]
            # assert filter_mask.shape[0] == x.shape[0]
            # assert filter_mask.shape[0] == grid.shape[0]
            x = x[filter_mask] #x[np,6=4(xyxy)+1(conf)+1(cls)]->x[np2,6=4(xyxy)+1(conf)+1(cls)]
            grid = grid[filter_mask] #grid[np]->grid[np2]

        # Filter by class
        if classes is not None:
            grid = grid[(x[:, -1:] == torch.tensor(classes, device=x.device)).any(1)]
            x = x[(x[:, -1:] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if n > max_nms:  # excess boxes
            grid = grid[x[:, -2].argsort(descending=True)[:max_nms]]  # sort by confidence
            x = x[x[:, -2].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, -1:] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, -2]  # boxes[nt,4] (offset by class), scores[nt]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # i[nt_nms]
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]        
        # assert i.max() < boxes.shape[0]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights [i, n] [1, n]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output_indices[xi] = grid[i]
        x_nms = x[i] #x[i][np->np_nms,6=4(xyxy)+1(conf)+1(cls)] -> x_nms[np_nms,518=4(xyxy)+512(nembd)+1(conf)+1(cls)]
        
        if m is not None and names_vec is not None: #names_vec[nc,n_embd]
            assert TMax>0
            text_vec = x_nms[:,4:-2] #text_vec[np_nms,n_embd]
            assert text_vec.shape==(len(i),n_embd)
            text_vec = m.pred_text(text_vec) #text_vec[np_nms,TMax*n_embd] -> text_vec[np_nms,TMax*n_embd]
            text_vec = normalize_dim(text_vec[:,0],dim=-1)#text_vec[np,n_embd]->text_vec[np,n_embd]
            # 计算点积矩阵: [np, nc]
            dot_matrix = torch.matmul(text_vec, names_vec.T)  # 或 dot_matrix = text_vec @ names_vec.T
            assert dot_matrix.shape==(text_vec.shape[0],names_vec.shape[0])
            # 沿 nc 维度求最大值和索引: [np]
            max_values, max_indices = torch.max(dot_matrix, dim=-1)
            # 拼接为 shape=[np, 2]，注意要先 unsqueeze 再 cat
            text_max = torch.stack([max_values, max_indices.float()], dim=1) #text_max[np, 2(max, max_id)]
            assert text_max.shape==(x_nms.shape[0],2)
            # idyolo = x_nms[:,-1] #idyolo[np] ~ text_max[:,0][np]
            # idyolo_int = idyolo.to(torch.int32) #idyolo_int[np]
            # text_max_int = text_max[:, 1].to(torch.int32) #text_max_int[np]
            # texts_ids.append(text_max)
            if nc!=names_vec.shape[0]:
                assert nc==1
                x_nms[:,-1] = text_max[:,-1]
            # [np,520=4(xyxy)+512(n_embd) + 2(max,max_id) +1(conf)+1(cls)]
            output[xi] = torch.cat((x_nms[:,:-2],text_max,x_nms[:,-2:]),-1)
        else:
            assert TMax==0
            output[xi] = x_nms #x[np,518=4(box)+512(n_embd) + 1(conf) + 1(cls)] -> x[xi][np_nms,518]
    
    return output#output[b][nt,520=4(xyxy)+512(n_embd) + 2(max,max_id) +1(conf)+1(cls)]
def non_max_suppression_olv(prediction, conf_thres=0.6, iou_thres=0.45, multi_label=False, agnostic=False,
                            max_det=300, model=None, names_vec=None):
    # Runs Non-Maximum Suppression (NMS) on inference results
    # prediction: [b, ntotal, 4(box)+nc + TMax*n_embd]
    # Returns:
    #      list of detections, on (n,4(box)+512(n_embd)+1(conf)+1(cls)) tensor per image [xyxy, conf, cls]  
    # output indices  
    indices_grid = torch.arange(0,prediction.shape[1], device=prediction.device).long() #indices_grid[ntotal]

    TMax, n_embd = 1, 512 #for CLIP

    nc = prediction.shape[2] - TMax*n_embd - 4  # number of classes
    assert names_vec==None or names_vec.shape==(nc,n_embd)

    if isinstance(conf_thres, float) or len(conf_thres.shape)==0:
        conf_thres = torch.ones(nc, device=prediction.device) * conf_thres
    else:
        if isinstance(conf_thres,np.ndarray):
            conf_thres = torch.from_numpy(conf_thres)
        conf_thres = conf_thres.to(prediction.device)
    #
    m = model.get_module_byname('YoloText') if model is not None else None
    assert m is not None
    #
    assert TMax>0

    # Checks
    assert ((0 <= conf_thres) & (conf_thres <= 1)).all(), f'Invalid Confidence threshold [{conf_thres.min()}, {conf_thres.max()}], valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    max_wh = 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output_indices = [torch.zeros((0, 1), device=prediction.device)] * prediction.shape[0]
    
    for xi, x_text in enumerate(prediction):  # image index, image inference
        #prediction[b,ntotal,4+nc]->x_text[ntotal,4+nc + TMax*n_embd]
        text_vec = x_text[:,-TMax*n_embd:] #text_vec[ntotal,n_embd]
        ntotal = text_vec.shape[0] #ntotal
        assert text_vec.shape==(ntotal,n_embd)
        text_vec = m.pred_text(text_vec) #text_vec[ntotal,TMax*n_embd] -> text_vec[ntotal,TMax*n_embd]
        text_vec = normalize_dim(text_vec[:,0],dim=-1)#text_vec[ntotal,n_embd]->text_vec[ntotal,n_embd]
        #
        # 计算点积矩阵: [np, nc]
        dot_matrix = torch.matmul(text_vec, names_vec.T) #->[ntotal,nc] 或 dot_matrix = text_vec @ names_vec.T
        assert dot_matrix.shape==(ntotal,nc)#names_vec.shape[0]
        xc = dot_matrix.max(-1)[0] > conf_thres.min() # candidates dot_matrix[ntotal,nc] -> xc[ntotal]
        if 0: # 沿 nc 维度求最大值和索引: [ntotal]
            max_values, max_indices = torch.max(dot_matrix, dim=-1) #max_values[ntotal],max_indices[ntotal]
            # 拼接为 shape=[ntotal, 2]，注意要先 unsqueeze 再 cat
            match_sim = torch.stack([max_values, max_indices.float()], dim=1) #text_max[ntotal, 2(max, max_id)]
            assert match_sim.shape==(ntotal,2)

        assert nc == prediction.shape[-1]-4-TMax*n_embd
            
        #xc[ntotal]->x_text[ntotal,4+nc + TMax*n_embd]
        # assert x_text.shape[0]==ntotal
        # assert indices_grid.shape[0]==ntotal
        x_text = x_text[xc]  # x_text[ntotal,4+nc + TMax*n_embd]->x_text[np,4+nc + TMax*n_embd]
        grid = indices_grid[xc] #indices_grid[ntotal]->grid[np]
        #
        x = x_text[...,:-TMax*n_embd] #x[np,4+nc]
        dot_matrix = dot_matrix[xc] #dot_matrix[ntotal,nc]->dot_matrix[np,nc]
        ptext = x_text[...,-TMax*n_embd:] #ptext[np,TMax*n_embd]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4]) ##x[np,533=4(xywh)+nc+nembd(512)]->box[np,4(xyxy)]
        # x[:,:4(xywh)] -> box[np,4(xyxy)]

        # dot_matrix_b[ntotal,nc]
        if multi_label:#
            i, j = (dot_matrix > conf_thres[None]).nonzero(as_tuple=False).T #i[np2]目标行号, j[np2]类列号
            nc_max, ncj = x[i,4:].max(1, keepdim=True) #x[np,nc]->nc_max[np,1], ncj[np,1]
            # assert torch.equal(x[i, 4 + j, None],nc_max[i,0])
            x = torch.cat((box[i], ptext[i], nc_max[i], ncj.float(), dot_matrix[i, j, None], j[:, None].float()), 1) #-> x[npi, 520 = 4(box)+512(n_embd) + 1(conf) + 1(cls) + 1(conf) + 1(cls)]
            grid = grid[i] #grid[npi]
        else:  # best class only
            conf, j = x[:, 4:].max(1, keepdim=True) #x[np,4+nc]->conf[np,1],j[np,1]
            filter_mask = (conf > conf_thres[j]).view(-1) #filter_mask[np]
            x = torch.cat((box, conf, j.float()), 1) #box[np,4(xyxy)],conf[np,1],j[np,1]->x[np,6=4(xyxy)+1(conf)+1(cls)]
            x = x[filter_mask] #x[np,6=4(xyxy)+1(conf)+1(cls)]->x[np2,6=4(xyxy)+1(conf)+1(cls)]
            grid = grid[filter_mask] #grid[np]->grid[np2]

        # Check shape
        n = x.shape[0]  # number of boxes
        if n > max_nms:  # excess boxes
            grid = grid[x[:, -2].argsort(descending=True)[:max_nms]]  # sort by confidence
            x = x[x[:, -2].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, -1:] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, -2]  # boxes[nt,4] (offset by class), scores[nt]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # i[nt_nms]
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]        
        # assert i.max() < boxes.shape[0]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights [i, n] [1, n]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output_indices[xi] = grid[i]
        x_nms = x[i] #x[i][np->np_nms,520=4(xyxy)+512(nembd)+1(conf)+1(cls)+1(conf)+1(cls)] -> x_nms[np_nms,520=4(xyxy)+512(nembd)+1(conf)+1(cls)+1(conf)+1(cls)]
        
        output[xi] = x_nms #x[np,520=4(xyxy)+512(nembd)+1(conf)+1(cls)+1(conf)+1(cls)]
    
    return output#output[b][nt,520=4(xyxy)+512(n_embd) + 2(max,max_id) +1(conf)+1(cls)]

def non_max_suppression_obb(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=1000, model=None, names_vec=None):
    # Runs Non-Maximum Suppression (NMS) on inference results
    # prediction: [batch, ntotal, 5(xywhr)+nc(16)+512(nembd)]
    # Returns:
    #      list of detections, on (n,7) tensor per image [xyxy, a, conf, cls]   
    # output indices  indices_grid[ntotal]
    indices_grid = torch.arange(0,prediction.shape[1], device=prediction.device).long()

    if names_vec is not None:
        TMax, n_embd = 1, 512 #for CLIP
    else:
        TMax, n_embd = 0, 0
    nc = prediction.shape[2] - TMax*n_embd - 5  # number of classes
    assert names_vec==None or (nc==1 and names_vec.shape[-1]==n_embd) or names_vec.shape==(nc,n_embd)

    if isinstance(conf_thres, float) or len(conf_thres.shape)==0:
        conf_thres = torch.ones(nc, device=prediction.device) * conf_thres
    else:
        if isinstance(conf_thres,np.ndarray):
            conf_thres = torch.from_numpy(conf_thres)
        conf_thres = conf_thres.to(prediction.device)
    nc_matrix = prediction[..., 5:-n_embd] if names_vec is not None else prediction[..., 5:] #nc_matrix[b,ntotal,nc]
    assert nc_matrix.shape[-1]==nc
    xc = nc_matrix.max(-1)[0] > conf_thres.min() # candidates nc_matrix[b,ntotal,nc] -> xc[b,ntotal]
    assert nc == prediction.shape[-1] - 5 -TMax*n_embd

    # Checks
    assert ((0 <= conf_thres) & (conf_thres <= 1)).all(), f'Invalid Confidence threshold [{conf_thres.min()}, {conf_thres.max()}], valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    max_wh = 7680  # (pixels) minimum and maximum box width and height
    max_nms = 6000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    m = model.get_module_byname('OBBText') if model is not None else None
    output = [torch.zeros((0, 7), device=prediction.device)] * prediction.shape[0]
    output_indices = [torch.zeros((0, 1), device=prediction.device)] * prediction.shape[0]
    for xi, x_text in enumerate(prediction):  # image index, image inference
        # x_text[ntotal,533=5(xywhr)+nc(16)+512]   x_text[nc,ntotal]
        # x_text[((x_text[..., 2:4] < min_wh) | (x_text[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # assert xc[xi].shape[0] == x_text.shape[0]
        # assert xc[xi].shape[0] == indices_grid.shape[0]
        x_text = x_text[xc[xi]] #x_text[ntotal,533=5(xywhr)+nc(16)+512(nembd)]->x_text[np,533=5(xywhr)+nc(16)+512]
        grid = indices_grid[xc[xi]] #indices_grid[ntotal]->grid[np]
        if TMax>0:
            x = x_text[...,:-TMax*n_embd] #x[np,5(xywhr)+nc]
            ptext = x_text[...,-TMax*n_embd:] #ptext[np,TMax*n_embd]
        else:
            x = x_text#x[np,5(xywhr)+nc]
            ptext = None

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :5] = l[:, 1:6]  # box
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :5] #x[np,533=5(xywhr)+nc+nembd]->box[np,5(xywhr)]
        # x[:,:4(xywh)] -> box[np,4(xyxy)]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:#
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T #i[np]目标行号, j[np]类列号
            if ptext is not None:
                x = torch.cat((box[i], ptext[i], x[i, j + 5, None], j[:, None].float()), 1) #x[npi,519=5(xywhr)+512(nembd) + 1(conf)+1(cls)]
            else:
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1) #x[npi,7=5(xywhr) + 1(conf)+1(cls)]
            grid = grid[i]
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True) #x[np,5(xywhr)+nc]->conf[np,1],j[np,1]
            filter_mask = (conf > conf_thres[j]).view(-1) #filter_mask[np]
            x = torch.cat((box, conf, j.float()), 1) # #box[np,5(xywhr)],conf[np,1],j[np,1]->x[np,519=5(xywhr)+1(conf)+1(cls)]
            # assert filter_mask.shape[0] == x.shape[0]
            # assert filter_mask.shape[0] == grid.shape[0]
            x = x[filter_mask] #x[np,519=5(xywhr)+512(nembd)+1(conf)+1(cls)]->x[np2,519=5(xywhr)+512(nembd)+1(conf)+1(cls)]
            grid = grid[filter_mask] #grid[np]->grid[np2]

        # Filter by class
        if classes is not None:
            grid = grid[(x[:, -1:] == torch.tensor(classes, device=x.device)).any(1)]
            x = x[(x[:, -1:] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # print(f'box4nms[{n}] > {max_nms}')
            filter_nms = x[:, -2].argsort(descending=True)[:max_nms]
            grid = grid[filter_nms]  # sort by confidence
            x = x[filter_nms]  # sort by confidence

        # Batched NMS
        c = x[:, -1:] * (0 if agnostic else max_wh)  # classes
        scores = x[:, -2]  # scores[np]
        boxes = torch.cat((x[:, :2] + c, x[:, 2:5]), dim=-1) #boxes[np,5]   xywhr offset

        # v5
        # polys2 = torch.cat([xywhr2xyxyxyxy(boxes).view(-1, 8), scores[:, None]], dim=-1).cpu().numpy().astype(np.float64)
        # i = poly_nms_cpu64(polys2, iou_thres)
        
        # v11
        i = nms_rotated(boxes, scores, iou_thres)

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]        
        # assert i.max() < boxes.shape[0]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = batch_probiou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights [i, n] [1, n]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output_indices[xi] = grid[i]
        x_nms = x[i] #x[i][np->np_nms,519=5(xywhr)+512(nembd)+1(conf)+1(cls)] -> x_nms[np_nms,519=5(xywhr)+512(nembd)+1(conf)+1(cls)]
        # if (time.time() - t) > time_limit:
        #     print(f'WARNING: NMS time limit {time_limit}s exceeded')
        #     break  # time limit exceeded
        if names_vec is not None: #names_vec[nc,n_embd]
            assert TMax>0
            text_vec = x_nms[:,5:-2] #text_vec[np_nms,n_embd]
            assert text_vec.shape==(len(i),n_embd)
            text_vec = m.pred_text(text_vec) #text_vec[np_nms,TMax*n_embd]->text_vec[np_nms,TMax*n_embd]
            text_vec = normalize_dim(text_vec[:,0],dim=-1)#text_vec[np,n_embd]->text_vec[np,n_embd]
            # 计算点积矩阵: [np, nc]
            dot_matrix = torch.matmul(text_vec, names_vec.T)  # 或 dot_matrix = text_vec @ names_vec.T
            assert dot_matrix.shape==(text_vec.shape[0],names_vec.shape[0])
            # 沿 nc 维度求最大值和索引: [np]
            max_values, max_indices = torch.max(dot_matrix, dim=-1)
            # 拼接为 shape=[np, 2]，注意要先 unsqueeze 再 cat
            text_max = torch.stack([max_values, max_indices.float()], dim=1) #text_max[np, 2(max, max_id)]
            assert text_max.shape==(x_nms.shape[0],2)
            # idyolo = x_nms[:,-1] #idyolo[np] ~ text_max[:,0][np]
            # idyolo_int = idyolo.to(torch.int32) #idyolo_int[np]
            # text_max_int = text_max[:, 1].to(torch.int32) #text_max_int[np]
            # texts_ids.append(text_max)
            if nc!=names_vec.shape[0]:
                assert nc==1
                x_nms[:,-1] = text_max[:,-1]
            # [np,521=5(xywhr)+512(n_embd) + 2(max,max_id) +1(conf)+1(cls)]
            output[xi] = torch.cat((x_nms[:,:-2],text_max,x_nms[:,-2:]),-1)
        else:
            assert TMax==0
            output[xi] = x_nms #output[xi][np,519=5(xywhr)+512(n_embd) + 1(conf) + 1(cls)]
    return output #output[b][np2,521=5(xywhr)+512(n_embd) + 2(max,max_id) + 1(conf)+1(c)]


def non_max_suppression_obb_no_nc(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=1000, model=None, names_vec=None, txt=True, no_nc=True):
    # Runs Non-Maximum Suppression (NMS) on inference results
    # prediction: [batch, ntotal, 5(xywhr)+nc(16)+512(nembd)]
    # Returns:
    #      list of detections, on (n,7) tensor per image [xyxy, a, conf, cls]   
    # output indices  indices_grid[ntotal]
    indices_grid = torch.arange(0,prediction.shape[1], device=prediction.device).long()

    if names_vec is not None:
        TMax, n_embd = 1, 512 #for CLIP
    else:
        TMax, n_embd = 0, 0
    nc = prediction.shape[2] - TMax*n_embd - 5  # number of classes
    # assert names_vec==None or (nc==1 and names_vec.shape[-1]==n_embd) or names_vec.shape==(nc,n_embd)

    # assert nc_matrix.shape[-1]==nc
    assert nc == prediction.shape[-1] - 5 -TMax*n_embd
    
    m = model.get_module_byname('OBBText') if model is not None else None
    if no_nc:
        text_vec = prediction[..., -TMax*n_embd:]
        text_vec_list = []
        for i in range(text_vec.shape[0]):
            text_vec_list.append(m.pred_text(text_vec[i]) if txt else text_vec[i])
        
        # text_vec = m.pred_text(text_vec) if txt else text_vec #text_vec[ntotal,TMax*n_embd] -> text_vec[ntotal,TMax*n_embd]
        text_vec = torch.stack(text_vec_list, dim=0).reshape_as(prediction[..., -TMax*n_embd:]) # [bs, ntotal, TMax*n_embd]

        # text_vec_mask = (text_vec * text_vec).sum(-1) < conf_thres
        text_vec = normalize_dim(text_vec,dim=-1)
        dot_matrix = torch.matmul(text_vec, names_vec.T) #dot_matrix[np,nembd]
        # dot_matrix[text_vec_mask] = 0
        prediction = torch.cat([prediction[..., :5], dot_matrix, prediction[..., -TMax*n_embd:]], dim=-1)
        nc = names_vec.shape[0]

    if isinstance(conf_thres, float) or len(conf_thres.shape)==0:
        conf_thres = torch.ones(nc, device=prediction.device) * conf_thres #  0.25 
    else:
        if isinstance(conf_thres,np.ndarray):
            conf_thres = torch.from_numpy(conf_thres)
        conf_thres = conf_thres.to(prediction.device)

    nc_matrix = prediction[..., 5:-n_embd] if names_vec is not None else prediction[..., 5:] #nc_matrix[b,ntotal,nc]
    xc = nc_matrix.max(-1)[0] > conf_thres.min() # candidates nc_matrix[b,ntotal,nc] -> xc[b,ntotal]
    # Checks
    assert ((0 <= conf_thres) & (conf_thres <= 1)).all(), f'Invalid Confidence threshold [{conf_thres.min()}, {conf_thres.max()}], valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    max_wh = 7680  # (pixels) minimum and maximum box width and height
    max_nms = 6000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 7), device=prediction.device)] * prediction.shape[0]
    output_indices = [torch.zeros((0, 1), device=prediction.device)] * prediction.shape[0]
    for xi, x_text in enumerate(prediction):  # image index, image inference
        # x_text[ntotal,533=5(xywhr)+nc(16)+512]   x_text[nc,ntotal]
        # x_text[((x_text[..., 2:4] < min_wh) | (x_text[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # assert xc[xi].shape[0] == x_text.shape[0]
        # assert xc[xi].shape[0] == indices_grid.shape[0]
        x_text = x_text[xc[xi]] #x_text[ntotal,533=5(xywhr)+nc(16)+512(nembd)]->x_text[np,533=5(xywhr)+nc(16)+512]
        grid = indices_grid[xc[xi]] #indices_grid[ntotal]->grid[np]
        if TMax>0:
            x = x_text[...,:-TMax*n_embd] #x[np,5(xywhr)+nc]
            ptext = x_text[...,-TMax*n_embd:] #ptext[np,TMax*n_embd]
        else:
            x = x_text#x[np,5(xywhr)+nc]
            ptext = None

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :5] = l[:, 1:6]  # box
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :5] #x[np,533=5(xywhr)+nc+nembd]->box[np,5(xywhr)]
        # x[:,:4(xywh)] -> box[np,4(xyxy)]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:#
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T #i[np]目标行号, j[np]类列号
            if ptext is not None:
                x = torch.cat((box[i], ptext[i], x[i, j + 5, None], j[:, None].float()), 1) #x[npi,519=5(xywhr)+512(nembd) + 1(conf)+1(cls)]
            else:
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1) #x[npi,7=5(xywhr) + 1(conf)+1(cls)]
            grid = grid[i]
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True) #x[np,5(xywhr)+nc]->conf[np,1],j[np,1]
            filter_mask = (conf > conf_thres[j]).view(-1) #filter_mask[np]
            if ptext is not None:
                x = torch.cat((box, ptext, conf, j.float()), 1) # #box[np,5(xywhr)],conf[np,1],j[np,1]->x[np,519=5(xywhr)+512(nembd)+1(conf)+1(cls)]
            else:
                x = torch.cat((box, conf, j.float()), 1)
            # assert filter_mask.shape[0] == x.shape[0]
            # assert filter_mask.shape[0] == grid.shape[0]
            x = x[filter_mask] #x[np,519=5(xywhr)+512(nembd)+1(conf)+1(cls)]->x[np2,519=5(xywhr)+512(nembd)+1(conf)+1(cls)]
            grid = grid[filter_mask] #grid[np]->grid[np2]

        # Filter by class
        if classes is not None:
            grid = grid[(x[:, -1:] == torch.tensor(classes, device=x.device)).any(1)]
            x = x[(x[:, -1:] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # print(f'box4nms[{n}] > {max_nms}')
            filter_nms = x[:, -2].argsort(descending=True)[:max_nms]
            grid = grid[filter_nms]  # sort by confidence
            x = x[filter_nms]  # sort by confidence

        # Batched NMS
        c = x[:, -1:] * (0 if agnostic else max_wh)  # classes
        scores = x[:, -2]  # scores[np]
        boxes = torch.cat((x[:, :2] + c, x[:, 2:5]), dim=-1) #boxes[np,5]   xywhr offset

        # v5
        # polys2 = torch.cat([xywhr2xyxyxyxy(boxes).view(-1, 8), scores[:, None]], dim=-1).cpu().numpy().astype(np.float64)
        # i = poly_nms_cpu64(polys2, iou_thres)
        
        # v11
        i = nms_rotated(boxes, scores, iou_thres)

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]        
        # assert i.max() < boxes.shape[0]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = batch_probiou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights [i, n] [1, n]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output_indices[xi] = grid[i]
        x_nms = x[i] #x[i][np->np_nms,519=5(xywhr)+512(nembd)+1(conf)+1(cls)] -> x_nms[np_nms,519=5(xywhr)+512(nembd)+1(conf)+1(cls)]
        # if (time.time() - t) > time_limit:
        #     print(f'WARNING: NMS time limit {time_limit}s exceeded')
        #     break  # time limit exceeded
        if names_vec is not None: #names_vec[nc,n_embd]
            assert TMax>0
            text_vec = x_nms[:,5:-2] #text_vec[np_nms,n_embd]
            assert text_vec.shape==(len(i),n_embd)
            text_vec = m.pred_text(text_vec) if txt else text_vec#text_vec[np_nms,TMax*n_embd]->text_vec[np_nms,TMax*n_embd]
            text_vec = text_vec.reshape_as(x_nms[:,5:-2])
            text_vec = normalize_dim(text_vec,dim=-1)#text_vec[np,n_embd]->text_vec[np,n_embd]
            # 计算点积矩阵: [np, nc]
            dot_matrix = torch.matmul(text_vec, names_vec.T)  # 或 dot_matrix = text_vec @ names_vec.T
            assert dot_matrix.shape==(text_vec.shape[0],names_vec.shape[0])
            # 沿 nc 维度求最大值和索引: [np]
            max_values, max_indices = torch.max(dot_matrix, dim=-1)
            # 拼接为 shape=[np, 2]，注意要先 unsqueeze 再 cat
            text_max = torch.stack([max_values, max_indices.float()], dim=1) #text_max[np, 2(max, max_id)]
            assert text_max.shape==(x_nms.shape[0],2)
            # idyolo = x_nms[:,-1] #idyolo[np] ~ text_max[:,0][np]
            # idyolo_int = idyolo.to(torch.int32) #idyolo_int[np]
            # text_max_int = text_max[:, 1].to(torch.int32) #text_max_int[np]
            # texts_ids.append(text_max)
            if nc!=names_vec.shape[0]:
                assert nc==1
                x_nms[:,-1] = text_max[:,-1]
            # [np,521=5(xywhr)+512(n_embd) + 2(max,max_id) +1(conf)+1(cls)]
            output[xi] = torch.cat((x_nms[:,:-2],text_max,x_nms[:,-2:]),-1)
        else:
            # assert TMax==0
            output[xi] = x_nms #output[xi][np,519=5(xywhr)+512(n_embd) + 1(conf) + 1(cls)]
    return output #output[b][np2,521=5(xywhr)+512(n_embd) + 2(max,max_id) + 1(conf)+1(c)]


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300, paths=None, imgsz=0):
    # Runs Non-Maximum Suppression (NMS) on inference results
    # Returns:
    #      list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    assert paths==None or len(paths)==prediction.shape[0]

    # prediction[b,nt,4(xywh)+1(obj)+cls] so prediction.shape[2]==cls
    nc = prediction.shape[2] - 5  # number of classes

    if isinstance(conf_thres, float):
        conf_thres = torch.ones(nc, device=prediction.device) * conf_thres
    else:
        conf_thres = conf_thres.to(prediction.device)
    xc = prediction[..., 4] > conf_thres.min()  # candidates
    
    # xc[b,nt]
    # xc[b,nt]

    # Checks
    assert ((0 <= conf_thres) & (conf_thres <= 1)).all(), f'Invalid Confidence threshold [{conf_thres.min()}, {conf_thres.max()}], valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 200.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    #box_array = []  # [n_imgs, n_boxes, xyxy=4 + obj_conf=1 + class=1 + mns_label=1]
    #for循环是对prediction[b,nt,4(xywh)+1(obj)+cls]0维度b进行循环
    for xi, x in enumerate(prediction):  # image index, image inference
        # xi=0,..,b-1  实际上这里推理b=1只有一次循环
        # x[nt,4(xywh)+1(obj)+cls] == prediction[xi=0,nt,4(xywh)+1(obj)+cls]
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # xc[1,nt]  # xc[xi] is [nt]
        x = x[xc[xi]]  # confidence过滤
        # x[np,4(xywh)+1(obj)+cls]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:#np
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # x[np,4(xywh)+1(obj)+cls]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # x[:, :4(xywh)] -> box[np,4(xyxy)]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:#同一个框可能对应好多个类，只要conf超过conf_thres
            i, j = (x[:, 5:] > conf_thres[None]).nonzero(as_tuple=False).T #i[np]目标行号, j[np]类列号
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float()), 1) #x[np,4(xyxy)+1(obj)+cls]->x[np2,4(xyxy)+1(conf)+1(cls)]
        else:  # best class only
            # x[np,4(xywh)+1(obj)+1(cls)]
            conf, j = x[:, 5:].max(1, keepdim=True) # #x[np,4(xywh)+1(obj)+nc]->conf[np,1],j[np,1] 选择class输出最大的conf作为最终可信度，j作为识别类标签
            # x[:, 5:]shape = [np,cls]
            # conf[np,1]
            # j[np,1]
            filter_mask = (conf > conf_thres[j]).view(-1) #filter_mask[np]
            x = torch.cat((box, conf, j.float()), 1)[filter_mask] #box[np,4(xyxy)],conf[np,1],j[np,1]->x[np,6=4(xyxy)+1(conf)+1(cls)]
            #->x[np2,6=4(xyxy)+1(conf)+1(cls)]  在1号维度拼起来[box+conf+cls]，再经过cls的conf过滤
        #x[np2,6=4(xyxy)+1(conf)+1(cls)]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms: # 砍掉可信度较低的,超过max_nms部分的boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            #x[:, 4].argsort(descending=True)[:max_nms]返回一个按[4]排序从大到小排序的整数id list

        # x[nobj_filt_cls,6==4(xyxy)+1(conf)+1(j.float()==cls)]
        # 上面是过滤,下面才是真正的nms

        # Batched NMS
        #x[np,4(xyxy)+1(conf)+1(cls)]
        c = x[:, 5:6]  # classes
        #c[np,1]
        scores = x[:, 4] #scores
        #

        if paths==None:
            if 0:
                boxes = x[:, :4]# boxes (offset by class)
                i_list=[]
                for ic in range(nc):
                    jc = (c==ic)
                    indices = torch.where(jc)[0]
                    # boxes[np,4]  scores[np]
                    idc = torchvision.ops.nms(boxes[indices], scores[indices], iou_thres)  # NMS
                    # idc[np] filt id
                    if idc.shape[0] > max_det:  # limit detections
                        idc = idc[:max_det]
                    # x[np, 6 == 4(xyxy) + 1(conf) + 1(cls)]
                    i_list.append(indices[idc])
                i = torch.cat(i_list,dim=0)
            else:
                boxes = x[:, :4] + c * (0 if agnostic else max_wh)   # boxes (offset by class)
                # boxes[np,4]  scores[np]
                i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
                # i[np] filt id
                if i.shape[0] > max_det:  # limit detections
                    i = i[:max_det]
                if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                    # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                    iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                    weights = iou * scores[None]  # box weights
                    x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                    if redundant:
                        i = i[iou.sum(1) > 1]  # require redundancy
            # x[np, 6 == 4(box) + 1(conf) + 1(cls)]
        else:
            boxes = x[:, :4]# boxes (offset by class)
            i_list=[]
            for ic in range(nc):
                jc = (c==ic)
                indices = torch.where(jc)[0]
                nt = indices.shape[0]
                if nt:
                    # boxes[np,4]  scores[np]
                    idc = torchvision.ops.nms(boxes[indices], scores[indices], iou_thres)  # NMS
                    # idc[np] filt id
                    if idc.shape[0] > max_det:  # limit detections
                        idc = idc[:max_det]
                    # x[np, 6 == 4(xyxy) + 1(conf) + 1(cls)]
                    idc0 = indices[idc]
                    i_list.append(idc0)
                    #
                    if 0:
                        # 保存boxes训练数据
                        nms_labels = torch.zeros(nt).to(x.device)   # x.shape[0] = n_boxes
                        nms_labels[idc] = 1  #nms_labels[nb]
                        #nms_labels = torch.unsqueeze(nms_labels, axis=1).to(x.device)#nms_labels[nb]->#nms_labels[nb,1]
                        #nms_labels = torch.tensor(nms_labels)
                        cx = (x[indices, 0] + x[indices, 2])/ 2 / imgsz
                        cy = (x[indices, 1] + x[indices, 3])/ 2 / imgsz
                        w  = (x[indices, 2] - x[indices, 0])/imgsz
                        h  = (x[indices, 3] - x[indices, 1])/imgsz
                        img_boxes_data = torch.cat((cx[:,None],cy[:,None],w[:,None],h[:,None], scores[indices,None], nms_labels[:,None]), dim=1)
                        #box_array.append(img_boxes_data)
                        file_name = os.path.basename(paths[xi]).rsplit('.', 1)[0]
                        save_path = './save_boxes/'
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        save_full_name = save_path + file_name + f'_{ic}' + '.txt'
                        np.savetxt(f'{save_full_name}', img_boxes_data.to('cpu').numpy(), fmt="%.4f %.4f %.4f %.4f %.4f %.d")
            i = torch.cat(i_list,dim=0)

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")


def print_mutation(results, hyp, save_dir, bucket):
    evolve_csv, results_csv, evolve_yaml = save_dir / 'evolve.csv', save_dir / 'results.csv', save_dir / 'hyp_evolve.yaml'
    keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
            'val/box_loss', 'val/cls_loss', 'val/dfl_loss') + tuple(hyp.keys())  # [results + hyps]
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)

    # Download (optional)
    if bucket:
        url = f'gs://{bucket}/evolve.csv'
        if gsutil_getsize(url) > (os.path.getsize(evolve_csv) if os.path.exists(evolve_csv) else 0):
            os.system(f'gsutil cp {url} {save_dir}')  # download evolve.csv if larger than local

    # Log to evolve.csv
    s = '' if evolve_csv.exists() else (('%20s,' * n % keys).rstrip(',') + '\n')  # add header
    with open(evolve_csv, 'a') as f:
        f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')

    # Print to screen
    print(colorstr('evolve: ') + ', '.join(f'{x.strip():>20s}' for x in keys))
    print(colorstr('evolve: ') + ', '.join(f'{x:20.5g}' for x in vals), end='\n\n\n')

    # Save yaml
    with open(evolve_yaml, 'w') as f:
        data = pd.read_csv(evolve_csv)
        data = data.rename(columns=lambda x: x.strip())  # strip keys
        i = np.argmax(fitness(data.values[:, :7]))  #
        f.write(f'# YOLOv5 Hyperparameter Evolution Results\n' +
                f'# Best generation: {i}\n' +
                f'# Last generation: {len(data)}\n' +
                f'# ' + ', '.join(f'{x.strip():>20s}' for x in keys[:7]) + '\n' +
                f'# ' + ', '.join(f'{x:>20.5g}' for x in data.values[i, :7]) + '\n\n')
        yaml.safe_dump(hyp, f, sort_keys=False)

    if bucket:
        os.system(f'gsutil cp {evolve_csv} {evolve_yaml} gs://{bucket}')  # upload


def apply_classifier(x, model, img, im0):
    # Apply a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('example%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        cv2.imwrite(str(increment_path(file, mkdir=True).with_suffix('.jpg')), crop)
    return crop


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def xywhr2xyxyxyxy(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in radians from 0 to pi/2.

    Args:
        x (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)

def xyxyxyxy2xywhr(x): #x[nt,8=4(pts)*2(xy)]
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
        rboxes.append([cx, cy, w, h, (angle % 360) / 180 * np.pi])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)


def nms_rotated(boxes, scores, threshold=0.45, use_triu=True):
    """
    NMS for oriented bounding boxes using probiou and fast-nms.

    Args:
        boxes (torch.Tensor): Rotated bounding boxes, shape (N, 5), format xywhr.
        scores (torch.Tensor): Confidence scores, shape (N,).
        threshold (float, optional): IoU threshold. Defaults to 0.45.
        use_triu (bool, optional): Whether to use `torch.triu` operator. It'd be useful for disable it
            when exporting obb models to some formats that do not support `torch.triu`.

    Returns:
        (torch.Tensor): Indices of boxes to keep after NMS.
    """
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes)
    if use_triu:
        ious = ious.triu_(diagonal=1)
        # pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
        # NOTE: handle the case when len(boxes) hence exportable by eliminating if-else condition
        pick = torch.nonzero((ious >= threshold).sum(0) <= 0).squeeze_(-1)
    else:
        n = boxes.shape[0]
        row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
        col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
        upper_mask = row_idx < col_idx
        ious = ious * upper_mask
        # Zeroing these scores ensures the additional indices would not affect the final results
        scores[~((ious >= threshold).sum(0) <= 0)] = 0
        # NOTE: return indices with fixed length to avoid TFLite reshape error
        pick = torch.topk(scores, scores.shape[0]).indices
    return sorted_idx[pick]


def check_amp(model, logger):
    """
    Checks the PyTorch Automatic Mixed Precision (AMP) functionality of a YOLO11 model. If the checks fail, it means
    there are anomalies with AMP on the system that may cause NaN losses or zero-mAP results, so AMP will be disabled
    during training.

    Args:
        model (nn.Module): A YOLO11 model instance.

    Example:
        ```python
        from ultralytics import YOLO
        from ultralytics.utils.checks import check_amp

        model = YOLO("yolo11n.pt").model.cuda()
        check_amp(model)
        ```

    Returns:
        (bool): Returns True if the AMP functionality works correctly with YOLO11 model, else False.
    """
    LOGGER = logger
    device = next(model.parameters()).device  # get model device
    prefix = colorstr("AMP: ")
    if device.type in {"cpu", "mps"}:
        return False  # AMP only used on CUDA devices
    else:
        # GPUs that have issues with AMP
        pattern = re.compile(
            r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)", re.IGNORECASE
        )

        gpu = torch.cuda.get_device_name(device)
        if bool(pattern.search(gpu)):
            LOGGER.warning(
                f"{prefix}checks failed ❌. AMP training on {gpu} GPU may cause "
                f"NaN losses or zero-mAP results, so AMP will be disabled during training."
            )
            return False

    def amp_allclose(m, im):
        """All close FP32 vs AMP results."""
        batch = im.repeat(2,1,1,1)
          # max stride P5-32 and P6-64
        a = m(batch)  # FP32 inference
        with autocast(enabled=True):
            b = m(batch) # AMP inference
        del m
        a = a[1] if isinstance(a, (tuple, list)) else a
        b = b[1] if isinstance(b, (tuple, list)) else b
        a = a[1] if isinstance(a, (tuple, list)) else a
        b = b[1] if isinstance(b, (tuple, list)) else b
        a = a[0, 0]
        b = b[0, 0]
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)  # close to 0.5 absolute tolerance
    if hasattr(model,'module'):
        if not hasattr(model,'stride'):
            model.stride = model.module.stride
        imgsz = max(256, int(model.stride.max() * 4))
    else:
        imgsz = 128
    im = torch.rand([1, 6, imgsz, imgsz], device=device)  # image to check
    LOGGER.info(f"{prefix}running Automatic Mixed Precision (AMP) checks...")
    warning_msg = "Setting 'amp=True'. If you experience zero-mAP or NaN losses you can disable AMP with amp=False."
    try:
        assert amp_allclose(model, im)
        LOGGER.info(f"{prefix}checks passed ✅")
    except ConnectionError:
        LOGGER.warning(
            f"{prefix}checks skipped ⚠️. Offline and unable to download YOLO11n for AMP checks. {warning_msg}"
        )
    except (AttributeError, ModuleNotFoundError):
        LOGGER.warning(
            f"{prefix}checks skipped ⚠️. "
            f"Unable to load YOLO11n for AMP checks due to possible Ultralytics package modifications. {warning_msg}"
        )
    except AssertionError:
        LOGGER.warning(
            f"{prefix}checks failed ❌. Anomalies were detected with AMP on your system that may lead to "
            f"NaN losses or zero-mAP results, so AMP will be disabled during training."
        )
        return False
    return True

def autocast(enabled: bool, device: str = "cuda"):
    """
    Get the appropriate autocast context manager based on PyTorch version and AMP setting.

    This function returns a context manager for automatic mixed precision (AMP) training that is compatible with both
    older and newer versions of PyTorch. It handles the differences in the autocast API between PyTorch versions.

    Args:
        enabled (bool): Whether to enable automatic mixed precision.
        device (str, optional): The device to use for autocast. Defaults to 'cuda'.

    Returns:
        (torch.amp.autocast): The appropriate autocast context manager.

    Note:
        - For PyTorch versions 1.13 and newer, it uses `torch.amp.autocast`.
        - For older versions, it uses `torch.cuda.autocast`.

    Example:
        ```python
        with autocast(amp=True):
            # Your mixed precision operations here
            pass
        ```
    """
    if TORCH_1_13:
        return torch.amp.autocast(device, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled)
    
def box_iou_liu(box1, box2):
    # box1: [N, 4] x,y,w,h
    # box2: [M, 4]
    # return: [N, M] iou
    # [x1,y1,x2,y2]
    box1_xy1 = box1[..., :2] - box1[..., 2:] / 2
    box1_xy2 = box1[..., :2] + box1[..., 2:] / 2
    box2_xy1 = box2[..., :2] - box2[..., 2:] / 2
    box2_xy2 = box2[..., :2] + box2[..., 2:] / 2

    inter_min = torch.max(box1_xy1[:, None, :], box2_xy1[None, :, :])
    inter_max = torch.min(box1_xy2[:, None, :], box2_xy2[None, :, :])
    inter = (inter_max - inter_min).clamp(min=0)  # [N,M,2]
    inter_area = inter[:, :, 0] * inter[:, :, 1]

    area1 = (box1_xy2 - box1_xy1).prod(1)
    area2 = (box2_xy2 - box2_xy1).prod(1)

    union = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / union.clamp(min=1e-6)
    return iou

def process_target_liu(target, B):
    # Process target tensor to convert OBB to HBB and create masks.
    # Args:
    #     target: [bnt, 10] for OBB (batch_idx, class, x1,y1,x2,y2,x3,y3,x4,y4)
    #             or [bnt, 6] for HBB (batch_idx, class, x,y,w,h)
    # Returns:
    #     processed_target: [B, nt_max, 6] (batch_idx, class, x,y,w,h)
    #     target_mask: [B, nt_max] boolean mask (True for valid targets)
    # Flatten all targets and separate batch indices
    batch_indices = target[:, 0].long()
    assert batch_indices.max() + 1 <= B
    
    # Convert OBB to HBB if needed
    if target.size(-1) == 10:
        # Get all points [bnt, 8]
        points = target[:, 2:].view(-1, 4, 2)
        
        # Find min/max to get HBB
        x_min = points[..., 0].min(dim=1)[0]
        x_max = points[..., 0].max(dim=1)[0]
        y_min = points[..., 1].min(dim=1)[0]
        y_max = points[..., 1].max(dim=1)[0]
        
        # Convert to xywh format
        x = x_min
        y = y_min
        w = x_max - x_min
        h = y_max - y_min
        
        # Create new target tensor [bnt, 6]
        processed_target = torch.stack([
            target[:, 0],  # batch_idx
            target[:, 1],  # class
            x, y, w, h
        ], dim=1)
    else:
        processed_target = target.clone()
    
    # Find max number of targets per batch
    counts = torch.bincount(batch_indices, minlength=B)
    nt_max = counts.max()
    
    # Initialize output tensors
    final_target = torch.zeros(B, nt_max, 6, device=target.device)
    target_mask = torch.zeros(B, nt_max, dtype=torch.bool, device=target.device)
    
    # Fill in the targets and mask
    for b in range(B):
        # Get targets for this batch
        mask = batch_indices == b
        batch_targets = processed_target[mask]
        
        if len(batch_targets) > 0:
            final_target[b, :len(batch_targets)] = batch_targets
            target_mask[b, :len(batch_targets)] = True
    
    return final_target, target_mask
