import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tqdm import tqdm
try:
    from modify import read_ablation_config, modify_df, process_files_and_configurations
except ImportError:
    # 这里写模块不存在时要执行的逻辑，也可以留空
    pass

#brow_path = r'/media/liu/f4854541-32b0-4d00-84a6-13d3a5dd30f2/workspace/pointnet/pts06-12-complex/coco2017ft'
#brow_path = r'D:\Articles\9DPose\experiments\NOSC'
#brow_path = r'D:\Articles\9DPose\experiments\linemode-voc'
#brow_path = r'D:\Articles\9DPose\experiments\kitti-2-cut=0_61.78'

#brow_path ='D:\Articles\9DPose\quanion\exp\plane9D'
#brow_path ='D:\Articles\9DPose\quanion\exp\kitti-2-cut=0_61.78'
#brow_path = r'D:\Articles\9DPose\quanion\exp\linemode-voc'

#brow_path = r'D:\Articles\9DPose\quanion\exp\NOCS' #L1
#brow_path = r'D:\Articles\9DPose\quanion\exp\carview' #L1
#brow_path = r'D:\Articles\9DPose\quanion\exp\kitti-2-cut=0_61.78' #L1
#brow_path = r'H:\shanxi\yolov5-box_can-kmean_fix-hsv\runs\train\bridge'
#brow_path = r'D:\Articles\remote\exp'
#brow_path = r'./checkpoint'
# brow_path = r'H:\python\Compare\results\kitti'
brow_path = r'runs/train'

if brow_path!='':
    # 根目录路径
    base_path = Path(brow_path)
    model_list = os.path.join(brow_path,'models.txt')
    if os.path.exists(model_list):
        subfolders = []
        with open(model_list, "r", encoding="utf-8") as f:
            for line in f:
                subfolders.append(os.path.join(base_path,line.strip()))  # 去除行尾的换行符和空白字符
    else:
        # 遍历base_path下的所有子文件夹
        subfolders = [str(f) for f in base_path.glob('*/') if f.is_dir()]
    # 将子文件夹路径转换为字符串列表
    file_paths = [folder for folder in subfolders if not '#' in folder and os.path.exists(str(Path(folder) / Path('results.csv')))]
else:
    file_paths = ['./results/kitti214/results.csv', './results/kitti215/results.csv']
methods=[]
for i,path in enumerate(file_paths):
    if Path(path).is_dir():
        path = os.path.join(path,'results.csv') #str(Path(path) / Path('results.csv'))
        file_paths[i] = path
        methods.append(Path(path).parent.name.strip())
assert len(methods)==len(file_paths)

# 加载CSV文件
dfs = [pd.read_csv(file_path) for file_path in file_paths if os.path.exists(file_path)]

# 确定有哪些技术指标（即列名），假设所有文件列名一致
tech_indicators = dfs[0].columns

# 定义需要展示的技术指标关键词列表
#selected_keywords = ['metrics/mAP_0.5', 'metrics/mAP_0.5:0.95']#['mAP', 'AP', 'f1', 'precision', 'recall', 'loss']
#selected_keywords = ['mAP', 'AP', 'f1', 'precision', 'recall']
selected_keywords = [] #all selected

selected_indicators = []
if len(selected_keywords)>0:
    # 按照selected_keywords顺序筛选和排序tech_indicators得到selected_indicators
    for keyword in selected_keywords:
        # 临时存储当前关键字匹配的指标，待验证其数据有效性
        temp_indicators = [indicator for indicator in tech_indicators if keyword in indicator]
        
        for indicator in temp_indicators:
            # 检查指标在所有dfs中是否至少有一个非空且非全零的数据集
            has_data = False
            for df in dfs:
                if indicator in df.columns and not df[indicator].isnull().all() and not (df[indicator] == 0).all():
                    has_data = True
                    break  # 找到有效数据，跳出循环
            
            # 如果指标有有效数据，则添加到selected_indicators
            if has_data and indicator not in selected_indicators:
                selected_indicators.append(indicator)
else:
    # temp_indicators = [indicator for indicator in tech_indicators if 'epoch' not in indicator.lower()]
    # 需要排除的关键词列表
    exclude_keywords = ['epoch', 'iter', 'time']
    # 过滤掉包含 exclude_keywords 中任意一个关键词的项
    temp_indicators = [
        indicator for indicator in tech_indicators
        if not any(keyword in indicator.lower() for keyword in exclude_keywords)
    ]
    for indicator in temp_indicators:
        has_data = False
        for df in dfs:
            if not df[indicator].isnull().all() and not (df[indicator] == 0).all():
                has_data = True
                break
        if has_data and indicator not in selected_indicators:
            selected_indicators.append(indicator)

# Assuming dfs and methods are already populated...
ablation_name = brow_path + '/ablation.txt'
if os.path.exists(ablation_name):
    configs = read_ablation_config(ablation_name)
    process_files_and_configurations(dfs, methods, configs, indes=['metrics/mAP_0.5', 'metrics/mAP_0.5:0.95'])

# 对每个技术指标绘制对比曲线图
# 计算最短和最长的epoch长度
min_length = min(len(df) for df in dfs)
max_length = max(len(df) for df in dfs)

# 确定最终的横轴长度
final_length = min(int(min_length * 4), max_length)

results_matrix = pd.DataFrame(index=methods, columns=selected_indicators)

# 对每个技术指标绘制对比曲线图
for iidc,indicator in enumerate(tqdm(selected_indicators)):
    plt.figure(figsize=(10, 6))
    best_value = None
    best_method = ''
    best_epoch = 0
    value_type = 'Max'
    best_color = 'black'  # 默认颜色，以防找不到最佳值对应的颜色
    for i, df in enumerate(dfs):
        # 绘制曲线，并获取曲线颜色
        if indicator in df.columns:
            line, = plt.plot(df[df.columns[0]], df[indicator], label=f'{methods[i]}')
            line_color = line.get_color()
            
            # 根据指标名称决定是寻找最大值还是最小值
            if 'loss' in indicator or 'error' in indicator:
                current_best = df[indicator].min()
                current_best_epoch = df[indicator].idxmin()
                value_type = 'Min'
            else:
                current_best = df[indicator].max()
                current_best_epoch = df[indicator].idxmax()
                value_type = 'Max'
            
            # Update the DataFrame with the current_best value
            results_matrix.at[methods[i], indicator] = current_best
            
            # 更新最佳值和对应信息
            if best_value is None or \
            (value_type == 'Max' and current_best > best_value) or \
            (value_type == 'Min' and current_best < best_value):
                best_value = current_best
                best_method = methods[i]
                best_epoch = current_best_epoch
                best_color = line_color  # 更新颜色
        else:
            print(f"\033[91mWarning: '{indicator}' not found in DataFrame{i}:{file_paths[i]}\033[0m")
        
    # 添加虚线网格
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # 标题、标签和图例
    plt.title(f'Comparison of {indicator.strip()} for {Path(brow_path).name}')
    plt.xlabel('Epoch')
    plt.ylabel(indicator)
    plt.legend()
    # 在图上标注最佳值，使用最佳曲线的颜色
    best_epoch2=df[df.columns[0]][0] + best_epoch
    plt.annotate(f'{value_type} by {best_method}: {best_value} at epoch {best_epoch2}',
                 xy=(best_epoch2, best_value), xytext=(best_epoch2, best_value),# + (best_value * 0.1 if value_type == 'Max' else -best_value * 0.1)
                 arrowprops=dict(facecolor=best_color, shrink=0.05),
                 color=best_color)
    plt.savefig(brow_path+'/'+indicator.strip().replace("/", "_").replace("\\", "_").replace(":", "-")+'.png')
    if iidc<2:
        plt.show()
    plt.close()

# Save the DataFrame to an Excel file
if brow_path!='':
    results_matrix.to_csv(Path(brow_path)/'indicator-methods.csv') #was to_excel

