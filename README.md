# SLFNet: A Stereo and LiDAR Fusion Network for Depth Completion

Pytorch implementation of "SLFNet: A Stereo and LiDAR Fusion Network for Depth Completion", RAL2022.

[[arXiv]](http://arxiv.org/abs/2009.08250)

## Overview
<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/PASMnet.png"/></div>

## Requirements
- Python 3.8
- PyTorch 1.6
- CUDA 10.2

## Train
### 1. Prepare training data
Download [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) and [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) datasets.

### 2. Train on SceneFlow
Run `./train.sh` to train on the SceneFlow dataset. Please update `datapath` in the bash file as your training data path.

### 3. Finetune on KITTI 2015
Run `./finetune.sh` to finetune on the KITTI 2015 dataset. Please update `datapath` in the bash file as your training data path.

## Test
### 1. Download pre-trained models
Download pre-trained models to `./log`.
- [Google Drive](https://drive.google.com/file/d/1_eXJnK8p-2NF4kxrj3ki6OHwXptO4iYp/view)
- [Baidu Drive](https://pan.baidu.com/s/1Yllm8992_n8i5YfwufyJ-Q)[code:fe12]

### 2. Test on SceneFlow
Run `./test.sh` to evaluate on the test set of the SceneFlow dataset. Please update `datapath` in the bash file as your test data path.

### 3. Test on KITTI 2015
Run `./submission.sh` to save png predictions on the test set of the KITTI 2015 dataset to the folder `./results`. Please update `datapath` in the bash file as your test data path.

## Results
<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Fig_PASMnet.png"/></div>

<img width="500" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Tab_PASMnet.png"/></div>

## Citation
```
@Article{zhang2022slfnet,
  author    = {Yongjian Zhang and Longguang Wang and Kunhong Li and Zhiheng Fu and Yulan Guo},
  title     = {{SLFNet}: A Stereo and LiDAR Fusion Network for Depth Completion},
  journal   = {{IEEE} Robotics and Automation Letters},
  year      = {2022},
}
```

## Acknowledgement

This code is built on [PAM](https://github.com/The-Learning-And-Vision-Atelier-LAVA/PAM) and [NLSPN](https://github.com/zzangjinsun/NLSPN_ECCV20). We thank the authors for sharing their codes.
# SLFNet-A-Stereo-and-LiDAR-Fusion-Network-for-Depth-Completion
# SLFNet-A-Stereo-and-LiDAR-Fusion-Network-for-Depth-Completion
