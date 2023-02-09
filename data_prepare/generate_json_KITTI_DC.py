"""
    毕业设计：基于双目相机和激光雷达的深度图像补全
    张勇健， 17308223， zhangyj85@mail2.sysu.edu.cn

    ======================================================================

    This script generates a json file for the KITTI Depth Completion dataset.
"""

import os
import argparse
import random
import json
import math

parser = argparse.ArgumentParser(
    description="KITTI Depth Completion jason generator")

# 注意修改参数，包括数据集根路径、输出路径及文件名；默认创建训练-验证-测试集。
parser.add_argument('--path_root', type=str, required=False,                    # KITTI补全数据集根目录
                    default='../../data_depth_velodyne',
                    help="Path to the KITTI Depth Completion dataset")
parser.add_argument('--path_out', type=str, required=False,                     # .json文件输出目录
                    default='../../data_json',
                    help="Output path")
parser.add_argument('--name_out', type=str, required=False,                     # .json文件命名
                    default='kitti_dc_train_val_test.json',
                    help="Output file name")
parser.add_argument('--num_train', type=int, required=False,                    # 训练数据样本最大值
                    default=int(1e10),
                    help="Maximum number of train data")
parser.add_argument('--num_val', type=int, required=False,                      # 验证集数据样本量
                    default=int(1e10),
                    help="Maximum number of val data")
parser.add_argument('--num_test', type=int, required=False,                     # 测试集数量
                    default=int(1000),
                    help="Maximum number of test data")
parser.add_argument('--seed', type=int, required=False,                         # 随机数种子
                    default=13,
                    help='Random seed')
parser.add_argument('--test_data', action='store_true',                         # 是否创建测试集
                    default=True,
                    help='json for DC test set generation')
args = parser.parse_args()

random.seed(args.seed)


# Some miscellaneous functions
def check_dir_existence(path_dir):
    assert os.path.isdir(path_dir), \
        "Directory does not exist : {}".format(path_dir)


def check_file_existence(path_file):
    assert os.path.isfile(path_file), \
        "File does not exist : {}".format(path_file)


def generate_json():
    
    # 生成包含 left + right + sparse + dense + K (train & val & test) 绝对路径的.json文件
    # 用于生成训练集和初始验证集
    
    check_dir_existence(args.path_out)              # 检查输出文件路径

    # For train/val splits
    dict_json = {}
    for split in ['train', 'val']:                  # 以train举例说明
        path_base = args.path_root + '/' + split    # 获取train绝对路径

        list_seq = os.listdir(path_base)            # 获取train目录下的所有文件名（场景序列）
        list_seq.sort()                             # 文件名排序

        list_pairs = []                             # 由每个scence，每个image_02/image_03中的　RGB+Depth+gt+K 组成的字典append的列表
        for seq in list_seq:
            cnt_seq = 0                             # scence计数

            # 获取场景序列list_seq中，image_02(左视图)对应的深度图的路径
            list_depth = os.listdir(path_base + '/' + seq + '/proj_depth/velodyne_raw/image_02')
            list_depth.sort()

            # 根据深度图，寻找对应的左右视图 &　pose & calib
            for name in list_depth:
                path_left = split + '/' + seq + '/' + 'image_02' + '/data/' + name                  # 左图路径  train/some_scence/image_02/data/xx.png
                path_right = split + '/' + seq + '/' + 'image_03' + '/data/' + name                 # 右图路径
                path_L_sparse = split + '/' + seq + '/proj_depth/velodyne_raw/image_02' + '/' + name  # 仅考虑左视图稀疏深度作为输入
                path_R_sparse = split + '/' + seq + '/proj_depth/velodyne_raw/image_03' + '/' + name
                path_dense = split + '/' + seq + '/proj_depth/groundtruth/image_02' + '/' + name    # 稠密深度图
                path_calib = split + '/' + seq + '/calib_cam_to_cam.txt'                            # 校准文件

                dict_sample = {
                    'left': path_left,
                    'right': path_right,
                    'L_sparse': path_L_sparse,
                    'R_sparse': path_R_sparse,
                    'dense': path_dense,
                    'K': path_calib
                }
                # 验证路径字典中的每个元素（value）是否存在
                flag_valid = True
                for val in dict_sample.values():
                    flag_valid &= os.path.exists(args.path_root + '/' + val)
                    if not flag_valid:
                        break

                if not flag_valid:
                    continue

                list_pairs.append(dict_sample)
                cnt_seq += 1

            print("{} : {} samples".format(seq, cnt_seq))

        dict_json[split] = list_pairs
        print("{} split : Total {} samples".format(split, len(list_pairs)))

    # 训练集与验证集字典加载完成，返回生成的字典
    return dict_json


def generate_json_test(dict_json):
    
    # For test split
    # 测试集从现有的验证集中产生，并将测试集样例从验证集中删除
    path_base = args.path_root + '/' + 'val'            # 验证集根路径

    list_pairs = []
    list_seq = os.listdir(path_base)

    # 统计验证集样本对总数
    val_pairs_num = len(dict_json['val'])

    # 获取从验证集中采样的比例
    sample_radio = args.num_test / val_pairs_num

    # 测试集划分
    for seq in list_seq:
        cnt_seq = 0

        # 某个sence下的所有深度图命名(如：0000000005.png)
        list_depth = os.listdir(path_base + '/' + seq + '/proj_depth/groundtruth/image_02')

        # 按照测试集和验证集之间的比例，在当前场景下随机筛选图像作为测试集数据，随后对筛选结果进行重新排
        sample_num = math.ceil(sample_radio * len(list_depth))
        random.shuffle(list_depth)
        list_depth = list_depth[:sample_num]
        list_depth.sort()

        # 根据深度图，寻找对应的左右视图 & pose & calib
        for name in list_depth:
            path_left   = 'val/' + seq + '/image_02' + '/data/' + name
            path_right  = 'val/' + seq + '/image_03' + '/data/' + name
            path_L_sparse = 'val/' + seq + '/proj_depth/velodyne_raw/image_02/' + name
            path_R_sparse = 'val/' + seq + '/proj_depth/velodyne_raw/image_03/' + name
            path_dense  = 'val/' + seq + '/proj_depth/groundtruth/image_02/' + name
            path_calib  = 'val/' + seq + '/calib_cam_to_cam.txt'

            dict_sample = {
                'left':   path_left,
                'right':  path_right,
                'L_sparse': path_L_sparse,
                'R_sparse': path_R_sparse,
                'dense':  path_dense,
                'K':      path_calib
                }

            # 验证路径字典中的每个路径元素是否存在
            flag_valid = True
            for val in dict_sample.values():
                flag_valid &= os.path.exists(args.path_root + '/' + val)
                if not flag_valid:
                    break

            if not flag_valid:
                continue

            list_pairs.append(dict_sample)
            cnt_seq += 1

        print("{} : {} samples".format(seq, cnt_seq))

    # 对所有场景进行筛选完毕后，需要确保选取的测试集满足预设要求，因此对超过该数量的数据集进行随机丢弃
    random.shuffle(list_pairs)
    list_pairs = list_pairs[:args.num_test]
    dict_json['test'] = list_pairs

    # 此外，在获取完测试集后，需要对验证集部分进行改写，将测试集样例从验证集中删除
    for i in range(len(dict_json['test'])):
        dict_json['val'].remove(dict_json['test'][i])

    print("Json file generation finished.")

    return dict_json


def post_process(dict_json):
    # 数据后处理
    # 打乱train数据集
    random.shuffle(dict_json['train'])

    # Cut if maximum is set
    for s in [('train', args.num_train), ('val', args.num_val)]:
        if len(dict_json[s[0]]) > s[1]:     # 若数据集大小超出给定大小
            # Do shuffle
            random.shuffle(dict_json[s[0]])

            num_orig = len(dict_json[s[0]])
            dict_json[s[0]] = dict_json[s[0]][0:s[1]]
            print("{} split : {} -> {}".format(s[0], num_orig,
                                               len(dict_json[s[0]])))

    f = open(args.path_out + '/' + args.name_out, 'w')
    json.dump(dict_json, f, indent=4)       # 缩进 indent = 4
    f.close()
    
    # 总结最终生成的文件数目
    print("train set size : {}" .format(len(dict_json['train'])))
    print("val set size : {}"   .format(len(dict_json['val'])))
    
    if args.test_data:
        print("test set size : {}"  .format(len(dict_json['test'])))
        
    print("Json file generation finished.")
    

if __name__ == '__main__':
    print('\nArguments :')
    for arg in vars(args):
        print(arg, ':',  getattr(args, arg))
    print('')

    # 生成训练集和初始测试集
    dict_json = generate_json()

    if args.test_data:
        dict_json = generate_json_test(dict_json)

    post_process(dict_json)
