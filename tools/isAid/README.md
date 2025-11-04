遥感领域第一个大型实例分割数据集，包含15类，共655,451个目标实例，图像数量达到2,806张。数据集中的目标类别涵盖了城市遥感解译的关键目标，如飞机、船只、车辆等，且标注精细，样本量大。

iSAID/
├── train/
│   ├── images/                               # 原始图像
│   │   ├── P0001.png
│   │   ├── P0002.png
│   │   └── ...
│   ├── Instance_masks/                       # 实例掩码图（RGB）
│   │   └── images/
│   │       ├── P0001_instance_id_RGB.png
│   │       ├── P0002_instance_id_RGB.png
│   │       └── ...
│   └── Annotations/
│       └── iSAID_train_2019.json             # COCO 格式标注文件（建议使用此文件）
│
├── val/
│   ├── images/
│   │   ├── P0003.png
│   │   ├── P0004.png
│   │   └── ...
│   ├── Instance_masks/
│   │   └── images/
│   │       ├── P0003_instance_id_RGB.png
│   │       ├── P0004_instance_id_RGB.png
│   │       └── ...
│   └── Annotations/
│       └── iSAID_val_2019.json               # COCO 格式标注文件（建议使用此文件）
│
├── AID2yolo.py                                # 编写的转换脚本（可选）
└── README.md                                  # 数据集说明文件（若有）


color_to_class = {
    (0, 0, 0): -1,  # 背景（黑色），跳过
    (0, 0, 63): 0,  # 类别ship
    (0, 63, 63): 1,  # storage tank
    (0, 63, 0): 2,  # baseball diamond
    (0, 63, 127): 3,  # tennis court
    (0, 63, 191): 4,  # basketball court
    (0, 63, 255): 5,  # ground track field
    (0, 127, 63): 6,  # bridge
    (0, 127, 127): 7,  # large vehicle
    (0, 0, 127): 8,  # small vehicle
    (0, 0, 191): 9,  # helicopter
    (0, 0, 255): 10,  # swimming pool
    (0, 191, 127): 11,  # roundabout
    (0, 127, 191): 12,  # soccer ball field
    (0, 127, 255): 13,  # plane
    (0, 100, 155): 14,  # harbor
}

{
  "images": [         // 图像信息
    {
      "id": 1,
      "file_name": "P0001.png"
    },
    ...
  ],
  "annotations": [    // 目标实例信息
    {
      "id": 1001,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x_min, y_min, width, height],
      "segmentation": [[x1,y1,x2,y2,...]],   // 多边形坐标
      "area": 1543.8,
      "iscrowd": 0
    },
    ...
  ],
  "categories": [     // 类别信息（固定15类）
    {"id": 1, "name": "plane"},
    {"id": 2, "name": "ship"},
    ...
  ]
}

用途	推荐路径	说明
检测任务（YOLO）	train/images/ + iSAID_train_2019.json	提 bbox 与 class_id
实例分割任务	train/Instance_masks/ + category.json	提取轮廓、颜色映射实例 ID
融合使用	同时使用 segmentation 和掩码图	可做验证或纠错