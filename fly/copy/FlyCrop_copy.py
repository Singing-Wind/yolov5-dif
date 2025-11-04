import cv2
import numpy as np
import math
import random
import torch
import os
from pygame.locals import *
import sys
# Placeholder for the predict function
# from predict import predict
from math2.transform import A232rot_torch,xcycdudv2A23_torch,A23inverse,apply_inverse_affine_and_map,cp_normalize
from general.config import load_config
from image.tif import read_tif_with_tifffile
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import kornia

class FlyCrop:
    def __init__(self,max_spd=60,fly_base=None):
        self.size_grab = (416, 416)
        self.hit_time = 0
        self.edge = 200
        self.spd_range = (max_spd/2,max_spd)
        self.fly_speed = (self.spd_range[0] + self.spd_range[1]) / 2
        self.scale = 1.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mA = torch.zeros((1, 2, 3), dtype=torch.float32).to(self.device)
        self.map = [None] if fly_base is None else fly_base.map
        self.map_t = [None] if fly_base is None else fly_base.map_t
        self.pos = torch.tensor([0.0, 0.0], dtype=torch.float32).to(self.device)
        self.v = torch.tensor([0.0, 0.0], dtype=torch.float32).to(self.device)
    
    def init_params(self,center=0):
        size = self.map[0].shape[1::-1]
        if center:
            self.pos[0] = size[0] / 2
            self.pos[1] = size[1] / 2
        else:
            margin = 20
            self.pos[0] = random.uniform(margin, size[0]-margin)
            self.pos[1] = random.uniform(margin, size[1]-margin)
        self.fly_speed = (self.spd_range[0] + self.spd_range[1]) / 2
        th = random.uniform(-math.pi, math.pi)
        self.v[0] = math.cos(th)
        self.v[1] = math.sin(th)
        self.scale = 1.0
        self.hit_time = 0
        cp_normalize(self.v, 2)
    
    def reload(self, map_name):
        if os.path.splitext(map_name)[1].lower()=='.tif':
            self.map[0] = read_tif_with_tifffile(map_name)
        else:
            self.map[0] = cv2.imread(map_name, cv2.IMREAD_COLOR)
        if self.map[0] is None:
            raise ValueError(f"Failed to load image: {map_name}")
        self.init_params()  # (width, height)
        #
        self.map_t[0] = torch.from_numpy(self.map[0]).float()  # [H,W,C]
        self.map_t[0] = self.map_t[0].permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
        # 转移到 GPU 上
        self.map_t[0] = self.map_t[0].to(self.device)
    
    def set_size(self, size):
        self.size_grab = size
        self.edge = 0.5 * (self.size_grab[0] + self.size_grab[1]) / 2
    
    def get_affine(self):
        #cos,sin==()
        self.mA[0, 0, 0] = -self.v[1]
        self.mA[0, 0, 1] = self.v[0]
        self.mA[0, 0, 2] = self.size_grab[0] / 2 - self.scale * (-self.pos[0] * self.v[1] + self.pos[1] * self.v[0])
        self.mA[0, 1, 0] = -self.v[0]
        self.mA[0, 1, 1] = -self.v[1]
        self.mA[0, 1, 2] = self.size_grab[1] / 2 - self.scale * (-self.pos[0] * self.v[0] - self.pos[1] * self.v[1])
    
    def render(self):
        if self.map[0] is None:
            return None
        frame = kornia.geometry.transform.warp_affine(self.map_t[0], self.mA, dsize=self.size_grab)
        #frame = cv2.warpAffine(self.map[0], self.mA, self.size_grab)
        return frame #frame[1,C,H,W]
    
    def bump(self):
        hit_edge = 0
        width, height = self.map[0].shape[1], self.map[0].shape[0]
        if (self.v[0] > 0 and self.pos[0] > width - self.edge) or (self.v[0] < 0 and self.pos[0] < self.edge):
            self.v[0] = -self.v[0]
            hit_edge |= 1
        if (self.v[1] > 0 and self.pos[1] > height - self.edge) or (self.v[1] < 0 and self.pos[1] < self.edge):
            self.v[1] = -self.v[1]
            hit_edge |= 2
        if hit_edge:
            self.hit_time += 1
        return hit_edge
    
    def rot(self, dth):
        vx, vy = self.v[0], self.v[1]
        dx, dy = math.cos(dth), math.sin(dth)
        self.v[0] = dx * vx - dy * vy
        self.v[1] = dy * vx + dx * vy
        cp_normalize(self.v, 2)
    
    def update_speed(self, dspd):
        self.fly_speed += random.uniform(-dspd, dspd)
        self.fly_speed = max(self.spd_range[0], min(self.fly_speed, self.spd_range[1]))
    
    def motion(self):
        self.pos += self.fly_speed * self.v