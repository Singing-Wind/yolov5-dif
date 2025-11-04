import cv2
# pip install opencv-python
# pip install opencv-contrib-python
import numpy as np
import torch
try:
    from osgeo import gdal, ogr, osr
except Exception as e:
    print('\033[91mLoad lib: osgeo-gdal fail\033[0]m') 
# sudo apt-get install libgdal-dev
# gdal-config --version ---->3.0.4  #(3.4.1)
# pip install pygdal==3.0.4.10  #(3.4.1.10)

#conda install -c conda-forge gdal
# ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 ~/anaconda3/envs/yolov5/lib/libffi.so.7
# https://github.com/ros-perception/vision_opencv/issues/509
# https://pypi.org/project/GDAL/
# https://gis.stackexchange.com/questions/28966/python-gdal-package-missing-header-file-when-installing-via-pip
# 首先'gdal库导入失败
# 代码中需要用到gdal库>:from osgeo import gdal
# 但是在运行时提示:No modulea named'osgeo’，即没有'osgeo'模块
# 于是 pipa install osgeo后再次运行，提示:
# cannot import name 'gdal 区
# 解决办法:
# whl下载地址:https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
# 根据自己的python版本选择对应的whl
# 我的python版本是3.6.5，
# 因此选择:GDAL-3.1.4-cp36-cp36m-win amd64.whl
# 下载完成后，保存，记住存储路径:
# pip install whl的存储路径 //记住一定是whl的存储路径，而不是pip install whl

def gdal_start(img_path):
    dataset = gdal.Open(img_path)
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    # im_bands = dataset.RasterCount  # 波段数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
    # im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    # im_proj = dataset.GetProjection()  # 获取投影信息
    # 16位转8位
    # 高， 宽
    # image = stretch_16to8(im_data)
    image = stretch_16to8_2(im_data) if len(im_data.shape)==2 else stretch_16to8_ch3(im_data) #im_data[C,H,W]->image[H,W,C]
    dataset = None
    # 拼接3通道
    if len(image.shape)==2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2) #image[H,W]->image[H,W,C]
    return image #image[H,W,C]

def stretch_16to8_2(bands, lower_percent=2, higher_percent=98):
    assert len(bands.shape)==2
    h, w = bands.shape
    a = 0  # np.min(band)
    b = 255  # np.max(band)
    c = np.percentile(bands, lower_percent)
    d = np.percentile(bands, higher_percent)
    rate = (b - a) / (d - c)
    bands = np.ravel(bands)
    size = bands.size // 2
    t1 = a + (bands[:size] - c) * rate
    t1[t1 < a] = a
    t1 = t1.astype(np.uint8)
    t2 = a + (bands[size:] - c) * rate
    t2[t2 > b] = b
    t2 = t2.astype(np.uint8)
    # out = np.concatenate((t1, t2)).reshape((h, w))
    bands[:size] = t1[:]
    bands[size:] = t2[:]
    return bands.reshape((h, w))
def stretch_16to8_ch3(bands, lower_percent=2, higher_percent=98, axis=2): #bands[C,H,W]->[h,w,3]
    assert len(bands.shape)==3
    ch0 = stretch_16to8_2(bands[0], lower_percent=lower_percent, higher_percent=higher_percent) #ch0[h,w]
    ch1 = stretch_16to8_2(bands[1], lower_percent=lower_percent, higher_percent=higher_percent) #ch1[h,w]
    ch2 = stretch_16to8_2(bands[2], lower_percent=lower_percent, higher_percent=higher_percent) #ch2[h,w]
    # 将 ch0, ch1, ch2 合并为 [3, h, w] 的数组
    merged_array = np.stack([ch2, ch1, ch0], axis=axis) #->merged_array[h,w,3]
    return merged_array #merged_array[h,w,3]
def stretch_16to8_bchw(bands, lower_percent=2, higher_percent=98): #bands[C,H,W]->[h,w,3]
    assert len(bands.shape)==4
    b, c, h, w = bands.shape
    merged_list = []
    for bands1 in bands:
        ch0 = stretch_16to8_2(bands1[0], lower_percent=lower_percent, higher_percent=higher_percent) #ch0[h,w]
        ch1 = stretch_16to8_2(bands1[1], lower_percent=lower_percent, higher_percent=higher_percent) #ch1[h,w]
        ch2 = stretch_16to8_2(bands1[2], lower_percent=lower_percent, higher_percent=higher_percent) #ch2[h,w]
        # 将 ch0, ch1, ch2 合并为 [3, h, w] 的数组
        merged1 = np.stack([ch2, ch1, ch0], axis=0) #->merged1[c,h,w]
        # merged_array = np.stack([merged_array,merged1], axis=0) #merged1[c,h,w]..->merged_array[b,c,h,w]
        # 将 merged1 添加到列表中
        merged_list.append(merged1)
    # 将列表中的数组合并为一个 [B, C, H, W] 的数组
    merged_array = np.stack(merged_list, axis=0)  # merged_array[B, C, H, W]
    return merged_array #merged_array[b,c,h,w]

def stretch_16to8_2_torch(bands, lower_percent=2, higher_percent=98):
    assert len(bands.shape) == 2
    h, w = bands.shape
    a = 0  # 输出最小值
    b = 255  # 输出最大值

    # 使用 PyTorch 计算百分位数
    c = torch.quantile(bands.float(), lower_percent / 100.0)
    d = torch.quantile(bands.float(), higher_percent / 100.0)
    # 拉伸公式
    rate = (b - a) / (d - c)
    bands_stretched = a + (bands - c) * rate
    # 截断超出范围的值
    bands_stretched = torch.clamp(bands_stretched, min=a, max=b)
    # 转换为 uint8 类型
    bands_stretched = bands_stretched.to(torch.uint8)
    # 恢复形状
    return bands_stretched.reshape((h, w))
def stretch_16to8_ch3_torch(bands, lower_percent=2, higher_percent=98, dim=2): #bands[C,H,W]->[h,w,3]
    assert len(bands.shape)==3
    ch0 = stretch_16to8_2_torch(bands[0], lower_percent=lower_percent, higher_percent=higher_percent) #ch0[h,w]
    ch1 = stretch_16to8_2_torch(bands[1], lower_percent=lower_percent, higher_percent=higher_percent) #ch1[h,w]
    ch2 = stretch_16to8_2_torch(bands[2], lower_percent=lower_percent, higher_percent=higher_percent) #ch2[h,w]
    # 将 ch0, ch1, ch2 合并为 [3, h, w] 的数组
    merged_array = torch.stack([ch2, ch1, ch0], dim=dim) #->merged_array[h,w,3]
    return merged_array #merged_array[h,w,3]
def stretch_16to8_bchw_torch(bands, lower_percent=2, higher_percent=98): #bands[C,H,W]->[h,w,3]
    assert len(bands.shape)==4
    b, c, h, w = bands.shape
    merged_list = []
    for bands1 in bands:
        ch0 = stretch_16to8_2_torch(bands1[0], lower_percent=lower_percent, higher_percent=higher_percent) #ch0[h,w]
        ch1 = stretch_16to8_2_torch(bands1[1], lower_percent=lower_percent, higher_percent=higher_percent) #ch1[h,w]
        ch2 = stretch_16to8_2_torch(bands1[2], lower_percent=lower_percent, higher_percent=higher_percent) #ch2[h,w]
        # 将 ch0, ch1, ch2 合并为 [3, h, w] 的数组
        merged1 = torch.stack([ch2, ch1, ch0], dim=0) #->merged1[c,h,w]
        # merged_array = np.stack([merged_array,merged1], axis=0) #merged1[c,h,w]..->merged_array[b,c,h,w]
        # 将 merged1 添加到列表中
        merged_list.append(merged1)
    # 将列表中的数组合并为一个 [B, C, H, W] 的数组
    merged_array = torch.stack(merged_list, axis=0)  # merged_array[B, C, H, W]
    return merged_array #merged_array[b,c,h,w]