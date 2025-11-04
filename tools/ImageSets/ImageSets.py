import os
import random
import yaml

def split_dataset(data_path, image_dir, train_txt, val_txt, p=0.8, seed=42):
    image_dir = os.path.join(data_path,image_dir)
    train_txt = os.path.join(data_path,train_txt)
    val_txt = os.path.join(data_path,val_txt)
    
    exts = ('.jpg', '.bmp', '.png', '.tif', '.tiff')
    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(exts)]

    random.seed(seed)
    random.shuffle(all_images)

    num_train = int(len(all_images) * p)
    train_images = all_images[:num_train]
    val_images = all_images[num_train:]

    with open(train_txt, 'w') as f_train:
        for img in train_images:
            f_train.write(f"./images/{img}\n")

    with open(val_txt, 'w') as f_val:
        for img in val_images:
            f_val.write(f"./images/{img}\n")

    print(f"✅ 数据划分完成：训练 {len(train_images)} 张，验证 {len(val_images)} 张")

def print_red(text):
    print(f"\033[91m{text}\033[0m")

def generate_yaml_config(folder_path):
    folder_path = os.path.abspath(folder_path)
    folder_name = os.path.basename(folder_path)

    # ✅ 判断 train.txt 和 val.txt 是否存在
    train_txt = os.path.join(folder_path, 'train.txt')
    val_txt = os.path.join(folder_path, 'val.txt')

    if not os.path.exists(train_txt):
        print_red(f'错误：未找到 {train_txt}')
        return

    if not os.path.exists(val_txt):
        print_red(f'错误：未找到 {val_txt}')
        return

    # ✅ 读取 names.txt
    names_file = os.path.join(folder_path, 'names.txt')
    names = []
    if os.path.exists(names_file):
        with open(names_file, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        print_red(f'错误：未找到 {names_file}')
        return

    # ✅ 构造 YAML 数据
    yaml_data = {
        'path': folder_path.replace('\\', '/'),  # Linux风格路径
        'train': 'train.txt',
        'val': 'val.txt',
    }

    # ✅ YAML 文件路径
    yaml_file_path = os.path.join(folder_path, f'{folder_name}.yaml')

    # ✅ 写入 YAML 文件
    with open(yaml_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)
        f.write(f"nc: {len(names)}\n")
        # 写 names 行内格式
        names_str = ", ".join([f"'{name}'" for name in names])
        f.write(f"names: [{names_str}]\n")
        f.write(
            '\n'
            '#ft_coef: 4\n'
            '#val_epoch: 10\n'
            '#val_epoch_T: 2\n'
            '# val_count: 200\n'
        )

    print(f"✔ 已生成 YAML 文件：{yaml_file_path}")

# 调用
if __name__ == '__main__':
    data_path = '/sgg/liujin/workspace/datas/S2Looking4-sim/train' #'/sgg/liujin/workspace/datas/WHU-BCD9/train'  /sgg/liujin/workspace/datas/S2Looking4-sim/train   /sgg/liujin/workspace/datas/CD_Data_GZ_Google8/train
    split_dataset(data_path,'images', 'train.txt', 'val.txt', p=0.9)
    generate_yaml_config(data_path)