"""
Finding the correspondance scene between KITTI raw data and Virtual KITTI
"""
import os

datapath = "/home/kunb/KITTI_DepthComplement/depth/data_depth_velodyne"
split = "train"   # [train, val]
rootpath = os.path.join(datapath, split)
scene_list = os.listdir(rootpath)
for scene in scene_list:
    path = os.path.join(rootpath, scene, 'image_02', 'data')
    if not os.path.isdir(path):
        continue
    file_list = os.listdir(path)
    if len(file_list) == 447:    # finding the same number list dir, [447, 233, 270, 339, 837]
        print(scene)
        exit()

print('Not Found!')