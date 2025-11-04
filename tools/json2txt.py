import json
import os
from tqdm import tqdm

root = 'E:/underwater1/dataset/UTDAC2020'
path = root + '/labels'

if not os.path.exists(path):
    os.mkdir(path)


# 加载COCO格式的JSON文件
with open('E:/underwater1/dataset/UTDAC2020/annotations/instances_val2017.json') as f:
    data = json.load(f)

# 假设我们已经知道了所有类别的映射关系
# 例如：
# 'echinus','starfish','holothurian','scallop'
#class_mapping = {'echinus': 0, 'starfish': 1, 'holothurian': 2, 'scallop': 3}
#class_mapping = {'echinus': 0, 'starfish': 1, 'holothurian': 2, 'scallop': 3}

classes = data['categories']#classes[{},{},{},....]
class_mapping = {}
for i,t in enumerate(classes):
    class_mapping[t['id']] = i
#classes = [t['id']=i for i,t in enumerate(classes)]
# 打开一个文件用于写入，如果文件不存在将被创建
with open(root+'/names.txt', 'w') as f:
    # 遍历列表中的每个元素
    for item in classes:
        # 将每个元素写入文件的一行，加上换行符
        f.write(f"{item['name']}\n")

max_id = -1
# 遍历所有的annotations
for ann in tqdm(data['annotations']):
    img_id = ann['image_id']
    category_id = ann['category_id']
    bbox = ann['bbox']

    # 获取这个annotation对应的图像信息，以获取图像尺寸
    img_info = next((item for item in data['images'] if item["id"] == img_id), None)
    img_width = img_info['width']
    img_height = img_info['height']

    # 转换COCO的bbox（[x_min, y_min, width, height]）为YOLO格式（[x_center, y_center, width, height]），并归一化
    x_center = (bbox[0] + bbox[2] / 2) / img_width
    y_center = (bbox[1] + bbox[3] / 2) / img_height
    width = bbox[2] / img_width
    height = bbox[3] / img_height
    
    # 将类别ID转换为你的类别索引
    class_index = class_mapping[category_id]#class_mapping[category_id]
    assert class_index>=0 and class_index<len(classes)
    if(class_index > max_id):
        max_id = class_index

    name, extension = os.path.splitext(img_info['file_name'])

    if bbox[2]>0 and width>0 and bbox[3]>0 and height>0:
        # 生成或追加到YOLO格式的标签文件
        label_file = os.path.join(path, f'{name}.txt')
        with open(label_file, 'a') as f:
            f.write(f'{class_index} {x_center} {y_center} {width} {height}\n')
    else:
        print(f'name={name}[{width}x{height}]')

class_number = max_id+1

print(f'class_number={class_number}')
