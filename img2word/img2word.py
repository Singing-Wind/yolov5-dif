import requests
import json
import base64
import numpy as np
import cv2
from image.image_proc import resize_to_max_area

def image_to_base64(image):
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            img_data = image_file.read()
    elif isinstance(image, np.ndarray):
        flag, img_data = cv2.imencode('.jpg', image)
        if not flag:
            raise ValueError('image can not be encoded into ".jpg"')
    else:
        raise ValueError('image not in [str, numpy.array]')
    encoded_string = base64.b64encode(img_data).decode('utf-8')
    return encoded_string

def img2sentence_qwen_vl(image,prompt='Describe this image in one sentence.',url = "http://10.255.255.1:8000/v1/chat/completions"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sgg12345."
    }
    
    payload = {
        "model": "/data/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct/",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            # "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                            "url": f"data:image/jpeg;base64,{image_to_base64(image)}"
                        }
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        # print("Response:")
        # print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        # if failed return None
        return None

def objimg2sentence_qwen_vl(image,obj_name, Chinese='', url = "http://10.255.255.1:8000/v1/chat/completions"): # in Chinese
    prompt = f'Describe the {obj_name} at the center of the image in one sentence{Chinese}, incorporating its surrounding environment.(without mentioning its central position in the image)'
    result = img2sentence_qwen_vl(image,prompt,url=url)
    return result
    
def extract_object_texts(image, keeps, names,Chinese='', url = "http://10.255.255.1:8000/v1/chat/completions"):
    # 从图像中根据keeps信息提取目标区域图像   
    # 参数:
    #     image: 原始图像(numpy数组)
    #     keeps: 目标检测结果[nt][10]或[nt][6]
    # 返回:
    #     obj_images: 提取的目标图像列表
    #     obj_infos: 目标信息列表(中心点坐标,宽高等)
    height, width = image.shape[:2]  # 获取原始图像宽高
    obj_texts = []
    
    for k in keeps:
        assert k.shape[-1] == 6 or k.shape[-1] == 10
        if k.shape[-1] == 10:
            # 处理4个点坐标的情况
            pts = k[:8].cpu().numpy().reshape(4, 2)  # 转换为4个点的坐标
            # 计算外接水平矩形框
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)       
        else:
            # 处理矩形框坐标的情况(类似处理，但直接从k中获取xyxy)
            assert k.shape[-1] == 6
            x1, y1, x2, y2 = k[:4].cpu().numpy()
            
        # 计算中心点和宽高
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        # 计算扩展后的区域(3倍宽高)
        new_w = 3 * w
        new_h = 3 * h
        
        # 计算裁剪区域坐标(处理边界情况)
        crop_x1 = max(0, int(cx - new_w / 2))
        crop_y1 = max(0, int(cy - new_h / 2))
        crop_x2 = min(width, int(cx + new_w / 2))
        crop_y2 = min(height, int(cy + new_h / 2))
        
        # 确保裁剪区域有效
        if crop_x2 > crop_x1 and crop_y2 > crop_y1:
            # 裁剪图像
            obj_img = image[crop_y1:crop_y2, crop_x1:crop_x2] #obj_img[H,W,C=3]
            obj_img = resize_to_max_area(obj_img,max_area=640*640)
            cls = k[-1].cpu()
            result = objimg2sentence_qwen_vl(obj_img,names[int(cls)],Chinese=Chinese,url=url)#obj_img[H,W,C=3]->result
            if result is not None:
                text = result['choices'][0]['message']['content'] #result->text
            else:
                text = ''

            obj_texts.append(text)
        else:
            obj_texts.append('')

    assert len(obj_texts)==len(keeps)
    
    return obj_texts

if __name__ == "__main__":
    # Case 1:
    img = cv2.imread('test.jpg')
    result = img2sentence_qwen_vl(img) 
    # print('Case 2:\n', result)
    text = result['choices'][0]['message']['content']
    print(text)

    # Case 2:
    result = img2sentence_qwen_vl('test.jpg') 
    print('Case 1:\n', result)
    print(result['choices'][0]['message']['content'])
