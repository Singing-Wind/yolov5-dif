import cv2

def resize_to_max_area(obj_img, max_area=640*640):
    # 将图像resize到最大面积不超过max_area，同时保持长宽比
    # 参数:
    #     obj_img: 输入图像，形状为[H,W,C=3]
    #     max_area: 目标最大面积（默认512×512）
    # 返回:
    #     resized_img: 缩放后的图像
    h, w = obj_img.shape[:2]
    current_area = h * w
    
    # 如果当前面积小于等于目标面积，直接返回原图
    if current_area <= max_area:
        return obj_img.copy()
    
    # 计算缩放比例（保持长宽比）
    scale_ratio = (max_area / current_area) ** 0.5  # 开平方根
    
    # 计算新尺寸（保持整数像素）
    new_w = int(w * scale_ratio)
    new_h = int(h * scale_ratio)
    
    # 确保至少1个像素
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    # 使用高质量插值方法进行缩放
    resized_img = cv2.resize(obj_img, (new_w, new_h), 
                           interpolation=cv2.INTER_AREA if scale_ratio < 1 else cv2.INTER_LINEAR)
    
    return resized_img