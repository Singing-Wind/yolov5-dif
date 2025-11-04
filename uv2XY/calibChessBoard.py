import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from glob import glob

# Paths
# data_path = r'H:\datas\dlsb417\total\road-20250212\train'
data_path = r'H:\datas\dlsb417\total\road0422'
IMAGE_DIR = os.path.join(data_path,"images")
CALIB_DIR = os.path.join(data_path,"calib")
os.makedirs(CALIB_DIR, exist_ok=True)

# Load image files
image_files = sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob(os.path.join(IMAGE_DIR, "*.png")))

# Parameters
TARGET_WIDTH, TARGET_HEIGHT = 896, 768
CIRCLE_RADIUS = 10  # Increased circle radius for selection

# Global Variables
current_index = 0
points = []  # List of ((u, v), (X, Y))
selected_point = (0,0)
sel = -1  # Selected point index
Hglob = None

# Tkinter root for input dialogs
root = tk.Tk()
root.withdraw()

def resize_image(image):
    h, w = image.shape[:2]
    scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h)), scale

def save_points():
    if not points:
        return
    img_name = os.path.basename(image_files[current_index])
    txt_path = os.path.join(CALIB_DIR, os.path.splitext(img_name)[0] + ".txt")
    with open(txt_path, "w") as f:
        for ((u, v), (X, Y)) in points:
            f.write(f"{u} {v} {X} {Y}\n")

def load_points():
    global points
    points = []
    img_name = os.path.basename(image_files[current_index])
    txt_path = os.path.join(CALIB_DIR, os.path.splitext(img_name)[0] + ".txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                u, v, X, Y = map(float, line.split())
                points.append(((u, v), (X, Y)))

def compute_homography():
    all_points = []
    for txt_file in glob(os.path.join(CALIB_DIR, "*.txt")):
        with open(txt_file, "r") as f:
            for line in f:
                u, v, X, Y = map(float, line.split())
                all_points.append(((u, v), (X, Y)))
    
    if len(all_points) >= 4:
        src_pts = np.array([p[0] for p in all_points], dtype=np.float32)
        dst_pts = np.array([p[1] for p in all_points], dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_pts) #, cv2.RANSAC
        if 1:
            dst_pts2 = cv2.perspectiveTransform(src_pts[None], H) #src_pts[1, n, 2]->dst_pts[1, n, 2]
            print(dst_pts2)
        np.save(os.path.join(CALIB_DIR, "homography.npy"), H)
        print("Homography matrix saved.")

import numpy as np
import cv2

def uv_to_ground_affine(all_points,K):
    # 输入：
    #     u, v: 图像坐标（可以是标量或 numpy 数组）
    #     H: 像素坐标到物体平面坐标的单应矩阵 (3x3)
    # 输出：
    #     x', y': 相机垂足坐标系下的坐标
    # Step 1: 将(u,v)转换为物体坐标(x,y)
    uv = np.array([p[0] for p in all_points], dtype=np.float32) #uv[n,2]
    xy = np.array([p[1] for p in all_points], dtype=np.float32) #xy[n,2]
    H, _ = cv2.findHomography(xy, uv) #, H[3,3]
    
    # uv_h = np.concatenate([uv, np.ones_like(uv[:,0:1])], axis=-1)  # shape (..., 3)
    # xy_h = (H @ uv_h.T).T  # shape (..., 3)
    # xy = xy_h[:, :2] / xy_h[:, 2:3]  # normalize

    # Step 2: 分解 H，得到 R 和 t（注意这里 H 是从物体坐标到图像坐标的变换，需求逆）
    # K = np.eye(3)  # 如果你有相机内参矩阵，替换这里
    H_norm = np.linalg.inv(K) @ H  # 归一化

    # 1. 计算缩放因子 lambda
    h1 = H_norm[:, 0]
    h2 = H_norm[:, 1]
    lambda_scale = (np.linalg.norm(h1) + np.linalg.norm(h2)) / 2.0
    # 2. 标准化
    H_norm /= lambda_scale

    r1 = H_norm[:, 0]
    r2 = H_norm[:, 1]
    t = H_norm[:, 2]
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    
    # 正交化 R（SVD 方法）
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    r1, r2, r3 = R[0,:], R[1,:], R[2,:]

    # Step 3: 计算相机光心在物体坐标系下的位置 T = -R^T @ t
    T = -R.T @ t
    Tx, Ty = T[0], T[1]

    # Step 4: 垂足坐标系轴
    y_axis = np.array([r3[0], r3[1]])
    x_axis = np.array([r3[1], -r3[0]])

    # 单位化
    y_axis /= np.linalg.norm(y_axis)
    x_axis /= np.linalg.norm(x_axis)

    A = np.stack([x_axis, y_axis], axis=0)  # 仿射矩阵 2x2

    # Step 5: 应用仿射变换
    delta = xy - np.array([Tx, Ty])
    xy_prime = (A @ delta.T).T

    return xy_prime[:, 0], xy_prime[:, 1], H

def make_camera_matrix(width, height, fk=1.2):
    # 生成相机内参矩阵 K。
    # 参数:
    #     width  (int or float):  图像宽度（像素）
    #     height (int or float):  图像高度（像素）
    #     fk     (float):         焦距相对于宽度的比例因子
    # 返回:
    #     K (3x3 ndarray): 相机内参矩阵
    fx = fk * width
    fy = fx          # 如果想用不同 fk 可以改这里，比如 fy = fk2 * height
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    return K
def compute_homography2(K):
    all_points = []
    for txt_file in glob(os.path.join(CALIB_DIR, "*.txt")):
        all_points1 = []
        with open(txt_file, "r") as f:
            for line in f:
                u, v, X, Y = map(float, line.split())
                all_points1.append(((u, v), (X, Y)))
        X, Y, H = uv_to_ground_affine(all_points1,K) #X[n], Y[n]
        for i,t in enumerate(all_points1):
            all_points.append((t[0], (X[i], Y[i])))
    
    if len(all_points) >= 4:
        src_pts = np.array([p[0] for p in all_points], dtype=np.float32)
        dst_pts = np.array([p[1] for p in all_points], dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_pts) #, cv2.RANSAC
        if 1:
            dst_pts2 = cv2.perspectiveTransform(src_pts[None], H) #src_pts[1, n, 2]->dst_pts[1, n, 2]
            print(dst_pts2)
        np.save(os.path.join(CALIB_DIR, "homography.npy"), H)
        print("Homography matrix saved.")
        return H
    else:
        return None

def mouse_callback(event, x, y, flags, param):
    global selected_point, sel
    scale = param
    selected_point = (x / scale,y / scale)
    if event == cv2.EVENT_LBUTTONDOWN:
        spt = selected_point
        xy_input = simpledialog.askstring("Input", "Enter X and Y (space separated):")
        if xy_input:
            try:
                X, Y = map(float, xy_input.split())
            except ValueError:
                X, Y = None, None
        
        if X is not None and Y is not None:
            points.append((spt, (X, Y)))
    elif event == cv2.EVENT_MOUSEMOVE:
        sel = -1  # Reset selection
        # u,v = x / scale, y / scale
        for i, ((ut, vt), _) in enumerate(points):
            if (x - ut*scale) ** 2 + (y - vt*scale) ** 2 < CIRCLE_RADIUS ** 2:
                sel = i
                break
import win32gui
import win32con
import win32api
import win32process
import win32ui
def force_english_ime(hwnd):
    # 1. 激活窗口
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass

    # 2. 发送切换输入法的消息
    english_hkl = 0x04090409
    win32gui.PostMessage(hwnd,
                         win32con.WM_INPUTLANGCHANGEREQUEST,
                         0,
                         english_hkl)

# 虚拟键码
VK_LCTRL = 0xA2
VK_RCTRL = 0xA3
def main():
    global current_index, sel, selected_point,Hglob
    window_name = "Calibration"
    cv2.namedWindow(window_name)
    # 获取窗口句柄
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        force_english_ime(hwnd)
    else:
        print(f"找不到窗口：{window_name}")
        return

    while True:
        img = cv2.imread(image_files[current_index])
        K = make_camera_matrix(img.shape[1],img.shape[0])
        img, scale = resize_image(img)
        load_points()

        while True:
            display = img.copy()
            for i, ((u, v), (X,Y)) in enumerate(points):
                us,vs = u*scale,v*scale
                color = (0, 255, 255) if i == sel else (0, 0, 255)
                cv2.circle(display, (int(us), int(vs)), CIRCLE_RADIUS, color, 2)
                cv2.drawMarker(display, (int(us), int(vs)), (255, 0, 0), cv2.MARKER_CROSS)
                # 显示坐标
                text_pos = (int(us) + 10, int(vs) - 10)
                cv2.putText(display, f'({X:.2f}, {Y:.2f})', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show mouse position and selection ID
            mouse_pos_text = f"{current_index}/{len(image_files)} {selected_point[0]:.1f},{selected_point[1]:.1f}"
            #cal homograph
            if Hglob is not None:
                uv = np.array([[selected_point]], dtype=np.float32) 
                dst_pts2 = cv2.perspectiveTransform(uv, Hglob) #src_pts[1, n, 2]->dst_pts[1, n, 2]
                u_prime, v_prime = dst_pts2[0, 0]
                mouse_pos_text += f'->{u_prime:.3f},{v_prime:.3f}'
            mouse_pos_text += f' sel={sel}'
            cv2.putText(display, mouse_pos_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            # if key!=255:
            #     print(key)

            if key == ord('s'):
                save_points()
            elif key == ord('h'):
                # compute_homography()
                Hglob = compute_homography2(K=K)
            elif key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                current_index = (current_index + 1) % len(image_files)
                break
            elif key == ord('m') and sel >= 0:
                tsel = sel
                xy_input = simpledialog.askstring("Input", "Enter X and Y (space separated):", initialvalue=f"{points[sel][1][0]} {points[sel][1][1]}")
                if xy_input:
                    try:
                        X, Y = map(float, xy_input.split())
                    except ValueError:
                        X, Y = None, None
                if X is not None and Y is not None:
                    uv = points[tsel][0]
                    points[tsel] = (uv, (X, Y))
            elif key in [ord('\b'), 46, ord('x')] and sel >= 0:  # Support both Backspace and DEL key
                points.pop(sel)
                sel = -1  # Reset selection
            elif (win32api.GetAsyncKeyState(VK_LCTRL) & 0x8000) or (win32api.GetAsyncKeyState(VK_RCTRL) & 0x8000):  # CTRL key
                current_index = (current_index - 1) % len(image_files)
                break
            cv2.setMouseCallback("Calibration", mouse_callback, scale)

if __name__ == "__main__":
    main()