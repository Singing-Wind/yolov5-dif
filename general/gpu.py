import torch

def get_gpu_info():
    # 获取所有可用 GPU 的信息，并将其格式化为字符串返回。
    # Returns:
    #     str: 包含所有 GPU 信息的字符串。
    gpu_info_str = ""

    # 检查是否有可用的 GPU
    if torch.cuda.is_available():
        # 获取 GPU 数量
        gpu_count = torch.cuda.device_count()
        gpu_info_str += f"gpu_count={gpu_count} "

        # 打印每个 GPU 的信息
        for i in range(gpu_count):
            gpu_info_str += f"{i}:{torch.cuda.get_device_name(i)}"
            gpu_info_str += f"[\033[92m{torch.cuda.memory_allocated(i) / 1e9:.2f}/{torch.cuda.get_device_properties(i).total_memory / 1e9:.2f}GB\033[0m]"
            gpu_info_str += f"Cached:{torch.cuda.memory_reserved(i) / 1e9:.2f}GB "
            gpu_info_str += f"CUDA Capability:{torch.cuda.get_device_capability(i)}"
    else:
        gpu_info_str = "No GPU available."

    return gpu_info_str

# 调用函数并打印结果
# gpu_info = get_gpu_info()
# print(gpu_info)