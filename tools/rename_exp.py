import os
import shutil
import yaml
import csv
import re
from pathlib import Path
import time

import psutil

def is_script_running(script_path: str) -> bool:
    script_path = os.path.abspath(script_path)
    for p in psutil.process_iter(attrs=["pid", "cmdline"]):
        try:
            cmdline = p.info["cmdline"]
            if not cmdline:
                continue
            # 一般 cmdline 形式是 ['python', 'train.py', ...]
            for arg in cmdline:
                if os.path.abspath(arg) == script_path:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def process_folder(root_dir,gap_time=0):
    is_training = is_script_running('train.py') if gap_time>0 else False
    for subfolder in os.listdir(root_dir):
        if not subfolder.startswith('exp'):
            continue
        
        full_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(full_path):
            continue
        
        opt_path = os.path.join(full_path, 'opt.yaml')
        hyp_path = os.path.join(full_path, 'hyp.yaml')
        if not os.path.isfile(opt_path) or not os.path.isfile(hyp_path):
            print(f"删除 {full_path}，因为缺少 opt.yaml 或 results.csv")
            shutil.rmtree(full_path)
            continue

        results_path = os.path.join(full_path, 'results.csv')
        weights_path_last = os.path.join(full_path, 'weights', 'last.pt')
        # 获取最后修改时间 (单位: 秒, 时间戳)
        mtime = os.path.getmtime(opt_path)
        if os.path.exists(results_path) and os.path.getmtime(results_path) > mtime:
            mtime = os.path.getmtime(results_path)
        if os.path.exists(weights_path_last) and os.path.getmtime(weights_path_last) > mtime:
            mtime = os.path.getmtime(weights_path_last)
        # 判断是否在 5 分钟以内
        if is_training and time.time() - mtime <= gap_time:  # 300 秒 = 5 分钟
            # print("文件在5分钟以内被修改过")
            continue

        weights_path = os.path.join(full_path, 'weights', 'best.pt')
        if not os.path.isfile(weights_path_last) and not os.path.isfile(weights_path)\
            or os.path.isfile(weights_path_last) and not os.path.exists(results_path):
            print(f"删除 {full_path}，因为没有best.pt or last.pt")
            shutil.rmtree(full_path)
            continue

        # 读取 opt.yaml
        with open(opt_path, 'r') as f:
            opt = yaml.safe_load(f)
        data_str = Path(opt.get('data', 'unknown')).stem
        cfg_str = Path(opt.get('cfg', 'unknown')).stem
        bs = opt.get('batch_size', 'unknown')
        bs_str = f'bs{bs}'
        # 读取 imgsz 的两个数字
        imgsz = opt.get('imgsz', [])  # 提供默认值以防缺失
        imgsz_str = f'-{imgsz[1]}x{imgsz[0]}' if len(imgsz)>0 else ''

        # 读取 results.csv
        with open(results_path, 'r', newline='') as f:
            reader = list(csv.DictReader(f))
            if not reader:
                print(f"删除 {full_path}，因为 results.csv 是空的")
                shutil.rmtree(full_path)
                continue
            # 清洗每一行的 key，移除字段名前后的空格
            reader_cleaned = [
                {k.strip(): v for k, v in row.items()}
                for row in reader
            ]
            try:
                # 找到 metrics/mAP_0.5 最大值所在的行
                best_row = max(reader_cleaned, key=lambda r: float(r.get('metrics/mAP_0.5', 0)))
                v1 = f"{(100*float(best_row['metrics/mAP_0.5'])):.2f}"
                v2 = f"{(100*float(best_row['metrics/mAP_0.5:0.95'])):.2f}"
                epo = reader_cleaned[-1]['epoch'].strip()
            except Exception as e:
                print(f"删除 {full_path}，因为 results.csv 解析失败: {e}")
                shutil.rmtree(full_path)
                continue

        # 提取 count
        match = re.match(r'exp(.+)$', subfolder)
        count_str = f'-{match.group(1)}' if match else ''

        # 构建新文件夹名
        folder_str = f'{data_str}-{cfg_str}-{bs_str}-{v1}-{v2}-epo{epo}{count_str}{imgsz_str}'
        new_folder_path = os.path.join(root_dir, folder_str)

        try:
            os.rename(full_path, new_folder_path)
            print(f"重命名 {subfolder} 为 {folder_str}")
        except Exception as e:
            print(f"重命名失败：{e}")

if __name__ == '__main__':
    root_directory = 'runs/train'  # <<< 修改为你的根目录路径
    process_folder(root_directory,gap_time=300)
