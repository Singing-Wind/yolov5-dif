import cv2
import numpy as np

def cal_overlay(W,w,lm):
    if W>w:
        nx_1 = (W-w) // (w-lm)
        if (W-w) % (w-lm) > 0:
            nx_1+=1
        return w - (W-w) / nx_1
    else:
        return lm

class CutImages:
    def __init__(self,  subsize=[640,640], over_lap=200, xyoff=[0,0], resize=0):
        self.subsize = subsize
        self.over_lap = over_lap
        self.xyoff = xyoff
        self.resize = resize

    def cut_images(self, image, imgsz):
        if isinstance(image, str):
            img_big = cv2.imread(image)  # BGR
        else:
            img_big = image
        if img_big is not None:
            # img_big = self.gdal_start(image_path)
            cut_affs = self._affine_imgs(img_big.shape[:2], imgsz) if (self.resize==0) else self._cut_imgs(img_big.shape[:2])
        else:
            cut_affs = [] 
        return img_big, cut_affs

    def _affine_imgs(self, img_shape, imgsz):
        height, width = img_shape#img.shape[:2]
        subsize = self.subsize
        #over_lap = self.over_lap
        #nx_1 = 1 + ( width-self.subsize[1]) // (self.subsize[1]-self.over_lap)
        #ny_1 = 1 + (height-self.subsize[0]) // (self.subsize[0]-self.over_lap)
        #self.subsize[1] - ( width-self.subsize[1]) / nx_1 if nx_1>0 else self.subsize[1]
        #self.subsize[0] - (height-self.subsize[0]) / ny_1 if ny_1>0 else self.subsize[0]
        self.over_lapxy = [cal_overlay(width,self.subsize[1],self.over_lap), cal_overlay(height,self.subsize[0],self.over_lap)]
        assert self.over_lapxy[0]>=self.over_lap and self.over_lapxy[1]>=self.over_lap

        cut_results = []
        ds = 0.5
        sx,sy = (imgsz[1]-ds) / (subsize[1]-ds), (imgsz[0]-ds) / (subsize[0]-ds)

        # 从左到右，从上到下
        dp = 0.1
        y = self.xyoff[1]
        while 1:
            x = self.xyoff[0]
            while 1:
                # x'=sx(x-x0) = [sx  0 -sx*x0]
                # y'=sy(y-y0) = [ 0 sy -sy*y0]
                # 计算仿射变换矩阵
                A23 = np.array([[sx, 0, -sx*(x - dp)], [0, sy, -sy*(y - dp)]], dtype=np.float32)
                #逆变换
                # x = x0 + x'/sx = [1/sx    0 x0]
                # y = y0 + y'/sy = [   0 1/sy y0]
                # 计算仿射变换矩阵
                A23_1 = np.array([[1.0/sx, 0, x-dp], [0, 1.0/sy, y-dp]], dtype=np.float32)
                #
                # 执行仿射变换
                #patch = cv2.warpAffine(img, A23, (int(imgsz[1]), int(imgsz[0])), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114,114,114))
                cut = {
                    'A23': A23,
                    'A23_1': A23_1,
                    #'patch': patch
                    'cxy': np.array([[1, 0, x+subsize[1]/2], [0, 1, y+subsize[0]/2]], dtype=np.float32)
                }
                cut_results.append(cut)
                if x + subsize[1] >= width:
                    break
                x += subsize[1] - self.over_lapxy[0]
            if y + subsize[0] >= height:
                break
            y += subsize[0] - self.over_lapxy[1]
        return cut_results
    
    def _cut_imgs(self, img_shape):
        height, width = img_shape#img.shape[:2]
        subsize = self.subsize
        over_lap = self.over_lap
        cut_results = []
        # 从左到右，从上到下
        x, y = 0, 0
        h_last = False
        while y < height:
            if y + subsize[0] >= height:
                y = height - subsize[0]
                h_last = True
            w_last = False
            x = 0
            while x < width:
                if x + subsize[1] >= width:
                    x = width - subsize[1]
                    w_last = True
                #patch = img[y:y + subsize[0], x:x + subsize[1]].copy()
                cut = {
                    'xy': [x, y],
                    #'patch': patch
                }
                cut_results.append(cut)
                if w_last:
                    break
                x += subsize[1] - over_lap
            if h_last:
                break
            y += subsize[0] - over_lap
        return cut_results

    
    # def gdal_start(self, img_path):
    #     dataset = gdal.Open(img_path)
    #     im_width = dataset.RasterXSize  # 栅格矩阵的列数
    #     im_height = dataset.RasterYSize  # 栅格矩阵的行数
    #     # im_bands = dataset.RasterCount  # 波段数
    #     im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
    #     # im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    #     # im_proj = dataset.GetProjection()  # 获取投影信息
    #     # 16位转8位
    #     # 高， 宽
    #     # image = self.stretch_16to8(im_data)
    #     image = self.stretch_16to8_2(im_data)
    #     dataset = None
    #     # 拼接3通道
    #     image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    #     return image

    def stretch_16to8_2(self, bands, lower_percent=2, higher_percent=98):
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