# SLFNet: A Stereo and LiDAR Fusion Network for Depth Completion

Pytorch implementation of "SLFNet: A Stereo and LiDAR Fusion Network for Depth Completion", RAL2022.

[[paper]](https://ieeexplore.ieee.org/abstract/document/9830848)


## Requirements
- Python 3.8
- PyTorch 1.6
- CUDA 10.2

## Usage
### 1. Prepare KITTI Depth Completion (KITTIDC) Dataset.
We used preprocessed KITTIDC dataset provided by [Jinsun Park](https://github.com/zzangjinsun/NLSPN_ECCV20).
- KITTI DC dataset is available at the [KITTI DC Website](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion).
- For color images, KITTI Raw dataset is also needed, which is available at the [KITTI Raw Website](https://www.cvlibs.net/datasets/kitti/raw_data.php).
- Please follow the official instructions (cf., devkit/readme.txt in each dataset) for preparation.

After downloading datasets, you should first copy color images, poses, and calibrations from the KITTI Raw to the KITTI DC dataset.
	$ cd SLFNet/data_prepare
	$ python prepare_KITTI_DC.py --path_root_dc PATH_TO_KITTI_DC --path_root_raw PATH_TO_KITTI_RAW
After that, you will get a data structure as follows:
	.
	├── depth_selection
	│    ├── test_depth_completion_anonymous
	│    │    ├── image
	│    │    ├── intrinsics
	│    │    └── velodyne_raw
	│    ├── test_depth_prediction_anonymous
	│    │    ├── image
	│    │    └── intrinsics
	│    └── val_selection_cropped
	│        ├── groundtruth_depth
	│        ├── image
	│        ├── intrinsics
	│        └── velodyne_raw
	├── train
	│    ├── 2011_09_26_drive_0001_sync
	│    │    ├── image_02
	│    │    │     └── data
	│    │    ├── image_03
	│    │    │     └── data
	│    │    ├── oxts
	│    │    │     └── data
	│    │    └── proj_depth
	│    │        ├── groundtruth
	│    │        └── velodyne_raw
	│    └── ...
	└── val
	    ├── 2011_09_26_drive_0002_sync
	    └── ...
After preparing the dataset, you should generate a json file containing paths to individual images.
	$ cd NLSPN_ROOT/utils

	# For Train / Validation
	$ python generate_json_KITTI_DC.py --path_root PATH_TO_KITTI_DC

	# For Online Evaluation Data
	$ python generate_json_KITTI_DC.py --path_root PATH_TO_KITTI_DC --name_out kitti_dc_test.json --test_data

### 2. Build and install DCN module for NLSPN.
	$ cd NLSPN_ROOT/src/model/deformconv
	$ sh make.sh
	
### 3. Training and Testing.
Modify the mode in the configure file, and:
	$ cd SLFNet
	$python main.py
Please refer to the SLFNet/configure/kitti_cfg.yaml for more options. The testing results will be save and you can analyze the results by
	$ python analyze.py

### 4. Pre-trained Model and Results
We release our pre-trained model on the [KITTIDC](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) dataset.
The pre-trained model is located in checkpoints/SLFNet/kitti.tar

## Results
|Dataset|RMSE(mm)|MAE(mm)|iRMSE(1/km)|iMAE(1/km)|
|----|----|----|----|----|
|KITTIDC|641.1|197.0|1.7727|0.8761|
|Virtual KITTI2|2843.16|696.2|6.794|2.007|

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
