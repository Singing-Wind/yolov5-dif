import os

def hbb_to_obb_pts(x, y, w, h):
    """将中心点和宽高转换为 OBB 四点坐标 (x1,y1,x2,y1,x2,y2,x1,y2)"""
    x1 = x - w / 2
    x2 = x + w / 2
    y1 = y - h / 2
    y2 = y + h / 2
    return [x1, y1, x2, y1, x2, y2, x1, y2]

def convert_labels_to_pts(label_dir):
    """将 label_dir 目录下的所有 .txt 标签文件转换为 .pts 文件"""
    for file_name in os.listdir(label_dir):
        if file_name.endswith('.txt'):
            txt_path = os.path.join(label_dir, file_name)
            pts_path = os.path.splitext(txt_path)[0] + '.pts'

            with open(txt_path, 'r') as f_in, open(pts_path, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # 跳过不合法行
                    cls_id, x, y, w, h = map(float, parts)
                    coords = hbb_to_obb_pts(x, y, w, h)
                    f_out.write(" ".join(f"{c:.6f}" for c in coords) + "\n")
    print("转换完成！")

if __name__ == "__main__":
    # 使用示例（假设 labels 文件夹在当前目录下）
    convert_labels_to_pts(r'/media/data4T/datas/SAR-AIR-SARShip/labels')
