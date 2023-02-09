"""
    毕业设计：基于双目相机和激光雷达的深度图像补全
    张勇健， 17308223， zhangyj85@mail2.sysu.edu.cn

    ======================================================================

    This script use for DataLoader.
    目前暂未输出标定文件，如果需要，则修改相关代码
"""


import os
import numpy as np
import json
import random
from . import BaseDataset

from PIL import Image
import torch
import torchvision.transforms.functional as TF


"""
KITTI Depth Completion json file has a following format:

{
    "train": [
        {
            "left": "train/2011_09_30_drive_0018_sync/image_02/data
                    /0000002698.png",
            "right": "train/2011_09_30_drive_0018_sync/image_03/data
                    /0000002698.png",
            "sparse": "train/2011_09_30_drive_0018_sync/proj_depth
                    /velodyne_raw/image_03/0000002698.png",
            "dense": "train/2011_09_30_drive_0018_sync/proj_depth/groundtruth
                    /image_03/0000002698.png",
            "K": "train/2011_09_30_drive_0018_sync/calib_cam_to_cam.txt"
        }, ...
    ],
    "val": [
        {
            "left": "val/2011_09_26_drive_0023_sync/image_02/data
                    /0000000218.png",
            "right": "val/2011_09_26_drive_0023_sync/image_03/data
                    /0000000218.png",
            "sparse": "val/2011_09_26_drive_0023_sync/proj_depth/velodyne_raw
                    /image_03/0000000218.png",
            "dense": "val/2011_09_26_drive_0023_sync/proj_depth/groundtruth
                    /image_03/0000000218.png",
            "K": "val/2011_09_26_drive_0023_sync/calib_cam_to_cam.txt"
        }, ...
    ]
}

Reference : https://github.com/XinJCheng/CSPN/blob/master/nyu_dataset_loader.py
"""

# 从png格式中获取深度，单位m
def read_depth(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))

    # Consider empty depth
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    # depth normalize to (1,256)
    image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth

# 从标定文件中获取内参和外参
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1) # 分割key:value
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

class KITTIDC(BaseDataset):
    def __init__(self, args, mode):
        super(KITTIDC, self).__init__(args, mode)

        self.args = args
        self.mode = mode                    # train / val / test

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        # 打开.json文件，读取每行路径
        with open(self.args["path"]["split_json"]) as json_file:
            json_data = json.load(json_file)        # json_data 为字典，包含train和val与test
            self.sample_list = json_data[mode]      # 提取所需的key和value

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # __getitem__()函数用于数据加载器DataLoader的枚举返回
        left, right, depthL, depthR, dense, K1, K2, K = self._load_data(idx)

        if self.mode in ['train']:                    # 对train数据进行截取，原图大小375 * 1242, 截取大小为256 * 1024
            width, height = left.size               # 获取图像大小

            crop_h = 256
            crop_w = 1024
            
            # 对图像进行顶部裁剪，左右随机裁剪
            h_start = height - crop_h
            w_start = (width  - crop_w) // 2 #random.randint(0, width - 1024)

            left   = TF.crop(left,   h_start, w_start, crop_h, crop_w)
            right  = TF.crop(right,  h_start, w_start, crop_h, crop_w)
            depthL = TF.crop(depthL, h_start, w_start, crop_h, crop_w)
            depthR = TF.crop(depthR, h_start, w_start, crop_h, crop_w)
            dense  = TF.crop(dense,  h_start, w_start, crop_h, crop_w)

            K1[0,2] -= w_start
            K1[1,2] -= h_start
            K2[0,2] -= w_start
            K2[1,2] -= h_start

        elif self.mode == 'val' or self.mode == 'test':

            pass

        else:
            # 其他数据，未实现其截取方案
            raise NotImplementedError

        # GuideNet 的归一化
        # left = np.array(left).astype(np.float32)
        # left = TF.normalize(TF.to_tensor(left), (90.995, 96.2278, 94.3213),
        #                      (79.2382, 80.5267, 82.1483), inplace=True)

        # ACMNet 直接将图像除以 255, PENet 直接采用 0-255 的数值
        # left = np.array(left).astype(np.float32) / 1
        # left = np.transpose(left, (2, 0, 1))
        # left = torch.tensor(left.copy())

        # 将双目图像载入tensor
        left  = TF.to_tensor(left)     # to_tensor(), PIL类 -> tensor类, 不改变数值范围
        # 依分布归一化到(-1,1), CCVN 不进行归一化
        left  = TF.normalize(left, (0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225), inplace=True)

        right  = TF.to_tensor(right)
        right = TF.normalize(right, (0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225), inplace=True)

        # 稀疏图装入tenser        
        depthL = TF.to_tensor(np.array(depthL))
        depthR = TF.to_tensor(np.array(depthR))
        dense  = TF.to_tensor(np.array(dense))
        
        # 按照基线和焦距，将深度转为视差
        L_valid_mask = depthL > 1e-8
        dispL = (K[0] * K[1]) / (depthL + 1e-8)
        dispL[~L_valid_mask] = 0.0                             # 将无效的深度像素位置零

        R_valid_mask = depthR > 1e-8
        dispR = (K[0] * K[1]) / (depthR + 1e-8)
        dispR[~R_valid_mask] = 0.0                             # 将无效的深度像素位置零

        output = {'rgb1': left, 'rgb2': right, 
                  'dep1': depthL, 'dep2': depthR,
                  'disp1': dispL, 'disp2':dispR,
                  'K1': K1, 'K2': K2,
                  'gt_depth': dense, 'K': K}

        return output

    def _load_data(self, idx):
        # 从.json文件中获取对应的路径
        path_left   = os.path.join(self.args["path"]["data_root"],
                                   self.sample_list[idx]['left'])
        path_right  = os.path.join(self.args["path"]["data_root"],
                                   self.sample_list[idx]['right'])
        path_L_sparse = os.path.join(self.args["path"]["data_root"],
                                     self.sample_list[idx]['L_sparse'])
        path_R_sparse = os.path.join(self.args["path"]["data_root"],
                                     self.sample_list[idx]['R_sparse'])
        path_dense  = os.path.join(self.args["path"]["data_root"],
                                   self.sample_list[idx]['dense'])
        path_cal    = os.path.join(self.args["path"]["data_root"],
                                   self.sample_list[idx]['K'])

        # 从标定文件中获取需要的数据
        calib = read_calib_file(path_cal)                   # 获取以{key,value}为组成的字典
        P_rect_02 = np.reshape(calib['P_rect_02'], (3, 4))  # 相机02的投影矩阵
        P_rect_03 = np.reshape(calib['P_rect_03'], (3, 4))  # 相机03的投影矩阵

        # 基线计算方法参考我的论文附录2，单位：m
        baseline = P_rect_02[0,3] / P_rect_02[0,0] - P_rect_03[0,3] / P_rect_03[0,0]

        # 获取相机焦距，单位：像素
        focal = P_rect_02[0,0]

        # 基线，焦距 转tensor
        K = torch.from_numpy(np.array([baseline, focal]).astype('float32'))
        K1 = torch.from_numpy(P_rect_02.astype('float32'))
        K2 = torch.from_numpy(P_rect_03.astype('float32'))
        
        # 载入双目图像
        left   = Image.open(path_left)
        right  = Image.open(path_right)

        # 深度值预处理，获取(0,1)范围内的结果
        depthL = read_depth(path_L_sparse)
        depthR = read_depth(path_R_sparse)
        dense  = read_depth(path_dense)

        # 格式转换
        # 模式“F”为32位浮点灰色图像，每个像素用32个bit表示，0表示黑，255表示白
        depthL = Image.fromarray(depthL.astype('float32'), mode='F')
        depthR = Image.fromarray(depthR.astype('float32'), mode='F')
        dense  = Image.fromarray(dense.astype('float32'),  mode='F')

        w1, h1 = left.size
        w2, h2 = right.size
        w3, h3 = depthL.size
        w4, h4 = dense.size

        assert w1 == w2 and w1 == w3 and w1 == w4 and h1 == h2 and h1 == h3 and h1 == h4

        return left, right, depthL, depthR, dense, K1, K2, K
