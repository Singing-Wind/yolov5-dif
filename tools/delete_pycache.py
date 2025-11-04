import os
import shutil

def delete_pycache(folder_path):
    # 递归删除文件夹下所有的 __pycache__ 文件夹及其内容
    # :param folder_path: 目标文件夹路径
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # 遍历当前目录下的所有文件和文件夹
        for dir_name in dirs:
            if dir_name == "__pycache__":
                # 找到 __pycache__ 文件夹
                pycache_path = os.path.join(root, dir_name)
                print(f"Deleting: {pycache_path}")
                # 删除 __pycache__ 文件夹及其内容
                shutil.rmtree(pycache_path)
        # 注意：topdown=False 确保先处理子文件夹，再处理父文件夹

if __name__ == "__main__":
    # 指定目标文件夹路径
    target_folder = './' #input('./').strip()

    # 检查路径是否存在
    if not os.path.isdir(target_folder):
        print(f"错误: 路径 '{target_folder}' 不存在或不是一个文件夹。")
    else:
        # 调用函数删除 __pycache__
        delete_pycache(target_folder)
        print("清理完成！")