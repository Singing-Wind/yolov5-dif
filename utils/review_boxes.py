#%%
import numpy as np
import torch 
import cv2
import pandas as pd
import os
import time
# from utils.general import xywh2xyxy

#%%
def save_objs(filepath, array):
    # array[batch_size=128, xyxy=4 + confidence=1 + class=20 + label=1 + prediction=1, boxes_size=199]
    os.makedirs(filepath, exist_ok=True)
    if type(array) == torch.Tensor:
        if array.is_cuda:
            tensor = tensor.cpu()
        np_data = array.numpy()
    elif type(array) == list:
        for i in array:
            if type(i) == torch.Tensor:
                if i.is_cuda:
                    i = i.cpu()
                np.savetxt(f'{filepath}/{time.time()}.txt', i.numpy(), fmt="%.4f, %.4f, %.4f, %.4f, %.4f, %.d, %.d")
        return
    elif type(array) == np.ndarray:
        np_data = array
    else:
        raise ValueError('Invalid array type')
    for i in range(np_data.shape[0]):
        np.savetxt(f'{filepath}/{time.time()}.txt', np_data[i][:,:].T)
    print(f'Saved {np_data.shape[0]} arrays to {filepath}')

def read_array(filepath):
    arrays = []
    for file in os.listdir(filepath):
        if file.endswith('.txt'):
            data = np.loadtxt(f'{filepath}/{file}')
            arrays.append(data)
    return np.array(arrays)

def draw(boxes, draw_class, image=None):
    if image is None: # 若不需要在外部图片上绘制
        image = np.zeros((640, 640, 3), dtype=np.uint8)
    height, width, _ =image.shape

    # 遍历每个检测框
    for box in boxes:
        xy1, xy2 = tuple(box[:2].astype(np.int32)), tuple(box[2:4].astype(np.int32))
        confidence = box[4]
        class_idx = box[5]
        label = box[6]
        if class_idx == draw_class:
            if box.shape[0] > 6:   # 若存在prediction
                color = (0, 0, 100+confidence*155) 
            else:
                color = (100+confidence*155, 100+confidence*155, 0)
            if label == 0:
                cv2.rectangle( image, xy1, xy2, color, 2) 
            else:
                cv2.rectangle( image, xy1, xy2, (0, 255, 0), 2)
    return image

def review(np_data, img=None):
    image_total = np_data.shape[0]
    class_total = 20 # TODO 
    show_image_idx = 0
    show_image_class = 0
    in_str = ''
    img_size = (640, 640) if img is None else img.shape[:2]
    while True:
        show_image_idx %= image_total
        show_image_class %= class_total
        image = draw(trans_date(np_data[show_image_idx], img_size), show_image_class)
        cv2.putText(image, f'Image: [{show_image_idx+1}/{image_total}], Class: [{show_image_class+1}/{class_total}]', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Input: {in_str}', (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('image', image)    
        key = cv2.waitKey(5)
        in_str = in_str.strip()
        if key == ord('\r'):
            if len(in_str) > 0:
                if in_str[0] == 'c' and len(in_str) > 1:
                    show_image_class = int(in_str[1:]) -1
                elif in_str[0] == 'i' and len(in_str) > 1:
                    show_image_idx = int(in_str[1:]) -1
                elif in_str[0] == 'q':
                    break
            in_str = ''
        elif key == 8:    # 若是退格键
            in_str = in_str[:-1]
        elif key == ord('k'):    # 上键
            show_image_idx += (image_total - 1)
        elif key == ord('j'):    # 下键
            show_image_idx += 1
        elif key == ord('h'):    # 左键
            show_image_class += (class_total -1)
        elif key == ord('l'):    # 右键
            show_image_class += 1
        elif key != -1 :
            in_str += chr(key)

    cv2.destroyAllWindows()
#%%
def trans_date(data, img_size):
    """
    将浮点xyxy和onehot_class数据, 转换为坐标+class数据
    Args:
        data: [n, 4 + conf=1 + cls=20 + other=?, 200]
        img_size: [height, width]
    """
    non_zeor_rows = np.any(data[:, :5] != 0, axis=1)
    data = data[non_zeor_rows]
    np_class = np.expand_dims(np.argmax(data[:, 5:25], 1), 1)
    xyxy = data[:, :4]
    xyxy[:, [0, 2]] *= img_size[1]
    xyxy[:, [1, 3]] *= img_size[0]
    return np.concatenate((xyxy, data[:, 4:5], np_class, data[:, 25:]), axis=1)

#%%
def draw_and_save(np_data, save_dir=None, img_path=None):
    if save_dir is None:
        save_dir = './output/'
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(np_data)):
        os.makedirs(f'{save_dir}/{i}', exist_ok=True)
        img_size = (640, 640) if img_path is None else (img_path[i].shape[0], img_path[i].shape[1])
        no_onehot_data = trans_date(np_data[i], img_size)
        df = pd.DataFrame(no_onehot_data)
        df.columns = ['x1', 'y1', 'x2', 'y2', 'confidence', 'class', 'label', 'prediction']
        conf_col = df['confidence']
        df = df.iloc[:, df.columns != 'confidence'] = df.iloc[:, df.columns != 'confidence'].astype(int)
        df['confidence'] = conf_col
        df.to_csv(f'{save_dir}/{i}/data.csv', index=False)
        for j in range(20): # TODO
            image = draw(no_onehot_data, j, img_path)
            cv2.imwrite(f'{save_dir}/{i}/{j}.jpg', image)

#%%
if __name__ == '__main__':
    # tensor_data = torch.randn(128, 26, 200)  # 对象张量
    # save_objs('./save_objs', tensor_data)

    np_data = read_array('./save_objs')
    # review(np_data)
    draw_and_save(np_data)
    # img = draw(np_data[1], 0, None)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# %%
