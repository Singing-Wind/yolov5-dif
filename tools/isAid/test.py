import cv2
import numpy as np
import os
import json
import re

def load_coco_annotations(annotations_path):
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def get_category_id_to_name(annotations):
    category_id_to_name = {}
    for category in annotations['categories']:
        category_id_to_name[category['id']] = category['name']
    return category_id_to_name

def get_image_id_to_file_name(annotations):
    image_id_to_file_name = {}
    for image in annotations['annotations']:
        image_id_to_file_name[image['id']] = image['file_name']
    return image_id_to_file_name

def get_annotations_by_image_id(annotations):
    annotations_by_image_id = {}
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = []
        annotations_by_image_id[image_id].append(annotation)
    return annotations_by_image_id

def mask_to_yolo(mask_path, image_path, output_dir, annotations_by_image_id, image_id_to_file_name, category_id_to_name):
    # 读取掩码图和原始图像
    mask = cv2.imread(mask_path)
    image = cv2.imread(image_path)
    if mask is None or image is None:
        print(f"Error: Unable to read mask or image file {mask_path} or {image_path}")
        return

    height, width, _ = image.shape

    # 找到掩码图中的所有唯一颜色值
    unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
    print(f"Unique colors in {mask_path}: {unique_colors}")

    # 获取当前图像的文件名
    imageTest = re.search(r'\d+', os.path.basename(mask_path).split('_')[0])
    image_file_name = image_id_to_file_name.get(int(imageTest.string), None)
    if image_file_name is None:
        print(f"Image file name not found for {mask_path}")
        return

    # 创建YOLO格式的标注文件
    output_file = os.path.join(output_dir, image_file_name.replace('.png', '.txt'))
    with open(output_file, 'w') as f:
        for color in unique_colors:
            if np.all(color == [0, 0, 0]):  # 跳过背景颜色
                continue

            # 创建一个掩码图，只包含当前颜色的像素
            instance_mask = np.all(mask == color, axis=2).astype(np.uint8) * 255

            # 找到当前实例的所有轮廓
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # 计算边界框
                x, y, w, h = cv2.boundingRect(contour)
                # 计算相对坐标
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                width_rel = w / width
                height_rel = h / height

                # 获取当前实例的类别标签
                instance_id = int(color[2])  # 假设实例ID存储在颜色值的B通道
                annotation = next((a for a in annotations_by_image_id.get(int(image_file_name.split('.')[0]), []) if a['id'] == instance_id), None)
                if annotation is None:
                    print(f"Annotation not found for instance ID {instance_id} in {mask_path}")
                    continue

                category_id = annotation['category_id']
                category_name = category_id_to_name[category_id]

                # 写入YOLO格式的标注信息
                f.write(f"{category_id - 1} {x_center:.6f} {y_center:.6f} {width_rel:.6f} {height_rel:.6f}\n")

def batch_convert(train_dir, output_dir, annotations_path):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载COCO格式的标注文件
    annotations = load_coco_annotations(annotations_path)
    category_id_to_name = get_category_id_to_name(annotations)
    image_id_to_file_name = get_image_id_to_file_name(annotations)
    annotations_by_image_id = get_annotations_by_image_id(annotations)

    # 遍历train目录中的所有掩码图文件
    masks_dir = os.path.join(train_dir, 'Instance_masks/images')
    for mask_file in os.listdir(masks_dir):
        if mask_file.endswith('_instance_id_RGB.png'):
            mask_path = os.path.join(masks_dir, mask_file)
            image_file = mask_file.replace('_instance_id_RGB.png', '.png')
            image_path = os.path.join(train_dir, 'images', image_file)

            if os.path.exists(image_path):
                mask_to_yolo(mask_path, image_path, output_dir, annotations_by_image_id, image_id_to_file_name, category_id_to_name)
                print(f"Processed {mask_file}")
            else:
                print(f"Image not found for {mask_file}")

# 主函数
if __name__ == "__main__":
    train_dir = 'train'  # 数据集的train目录路径
    output_dir = 'train/annotations'  # 输出YOLO格式标注文件的目录路径
    annotations_path = 'train/Annotations/iSAID_train_2019.json'  # COCO格式标注文件路径
    batch_convert(train_dir, output_dir, annotations_path)