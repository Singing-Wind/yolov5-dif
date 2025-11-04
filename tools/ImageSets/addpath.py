import os

def convert_names_to_paths(data_path, name, prefix="./images/", suffix=".jpg"):
    # 将输入文件中每行的名字转成路径格式，例如：
    # 原：  0_0_5
    # 变： ./images/0_0_5.jpg
    input_path = os.path.join(data_path,'ImageSets',name)
    output_path = os.path.join(data_path,name)
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            name = line.strip()
            if name:  # 跳过空行
                new_line = f"{prefix}{name}{suffix}\n"
                outfile.write(new_line)
    print(f"✅ 转换完成：{input_path} → {output_path}")

if __name__ == "__main__":
    # 调用两次处理 train 和 test
    convert_names_to_paths("K:/datas/rsdd", "train.txt")
    convert_names_to_paths("K:/datas/rsdd", "test.txt")