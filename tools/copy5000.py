import shutil
import os

# 设置源文件夹和目标文件夹
source_dir = './images33000'
target_dir = './images'

# 确保目标文件夹存在
os.makedirs(target_dir, exist_ok=True)

count = 0
# 打开包含文件名的文本文件
with open('1.txt', 'r') as file:
    for filename in file:
        filename = filename.strip()  # 去除可能的空格和换行符
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename)

        # 检查源文件是否存在
        if os.path.exists(source_file):
            if not os.path.exists(target_file):
                shutil.copy(source_file, target_file)
            count+=1
            if(count>=5000):
                break
        else:
            print(f"File {filename} not found in {source_dir}")
