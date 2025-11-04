import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from glob import glob

# Paths
data_path = r'H:\datas\dlsb417\total\road-20250212\train'
IMAGE_DIR = os.path.join(data_path,"images")
CALIB_DIR = os.path.join(data_path,"calib")
os.makedirs(CALIB_DIR, exist_ok=True)

# Load image files
image_files = sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob(os.path.join(IMAGE_DIR, "*.png")))

# Parameters
TARGET_WIDTH, TARGET_HEIGHT = 800, 600
CIRCLE_RADIUS = 10  # Increased circle radius for selection

# Global Variables
current_index = 0
points = []  # List of ((u, v), (X, Y))
selected_point = (0,0)
sel = -1  # Selected point index

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

def main():
    global current_index, sel, selected_point
    while True:
        img = cv2.imread(image_files[current_index])
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
            mouse_pos_text = f"Mouse:{selected_point[0]:.1f},{selected_point[1]:.1f} {sel}"
            cv2.putText(display, mouse_pos_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            # if key!=255:
            #     print(key)

            if key == ord('s'):
                save_points()
            elif key == ord('h'):
                compute_homography()
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
            elif key == ord('c') or (key == 17 and cv2.waitKey(0) == 17):  # CTRL key
                current_index = (current_index - 1) % len(image_files)
                break
            cv2.setMouseCallback("Calibration", mouse_callback, scale)

if __name__ == "__main__":
    main()