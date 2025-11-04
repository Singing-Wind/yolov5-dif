import cv2
import numpy as np

# bbox.hpts = rect2objpts(xywh,H=H)
# for i in range(xywh.shape[0]):
#     cx, cy, w, h = xywh[i]
#     hpts = bbox.hpts[i] #hpts[4,2]
#     L = average_distance(hpts)

def rect2objpts(xywh,H): #xywh[nt,4(xywh)]
    w,h = xywh[:,2],xywh[:,3] #w[nt],h[nt]
    x1,y1 = xywh[:,0] - w/2, xywh[:,1] - h/2
    x2,y2 = xywh[:,0] + w/2, xywh[:,1] + h/2
    # 形成 [n, 4, 2] 形状的 src_pts
    src_pts = np.stack([
        np.stack([x1, y1], axis=-1),  # 左上角
        np.stack([x2, y1], axis=-1),  # 右上角
        np.stack([x2, y2], axis=-1),  # 右下角
        np.stack([x1, y2], axis=-1)   # 左下角
    ], axis=1).astype(np.float32)  # 形状 src_pts[n, 4, 2]

    dst_pts = cv2.perspectiveTransform(src_pts, H) #src_pts[n, 4, 2]->dst_pts[n, 4, 2]

    return dst_pts #dst_pts[n, 4, 2]

def average_distance(hpts):
    # 计算 0 和 2 之间的距离
    d1 = np.linalg.norm(hpts[0] - hpts[2])
    
    # 计算 1 和 3 之间的距离
    d2 = np.linalg.norm(hpts[1] - hpts[3])
    
    # 计算平均值
    avg_dist = (d1 + d2) / 2
    return avg_dist


if __name__ == '__main__':
    # 定义源点和目标点(至少需要4对点)
    src_pts = np.array([
        [30.0, 50.0],
        [120.0, 50.0],
        [120.0, 160.0],
        [30.0, 160.0]
    ], dtype=np.float32)

    dst_pts = np.array([
        [50.0, 70.0],
        [160.0, 30.0],
        [150.0, 180.0],
        [40.0, 150.0]
    ], dtype=np.float32)

    # 计算Homography矩阵
    H, status = cv2.findHomography(src_pts, dst_pts)

    print("Homography Matrix:")
    print(H)

# 可以使用该矩阵进行图像变换
# warped_image = cv2.warpPerspective(src_image, H, (width, height))
# 注意事项
# 至少需要4对非共面的点来计算Homography矩阵
# 点对越多，计算结果越精确
# 对于可能存在异常值的情况，建议使用RANSAC或LMEDS方法
# 计算出的Homography矩阵可以用于cv2.warpPerspective()函数进行图像变换
# 通过这种方法，你可以轻松地在OpenCV中计算两个平面之间的Homography变换矩阵。