from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from PIL import Image
import torchvision.transforms.functional as TF
from utils import *

# 加载进度条, 可视化训练进度
# from tqdm import tqdm

# 用于加载数据集的函数
from dataset import kittidc as DA

############################## 从 configure 中导入参数配置 ##############################
import yaml
args = yaml.safe_load(open("./configure/kitti_cfg.yaml", 'r'))

# 配置 GPU
if args["environment"]["only_test"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = args["test"]["gpus_id"]
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args["train"]["gpus_id"]

# 启用 CUDA = (not x) and y，not 0 and 1时使能cuda
if not (args["environment"]["use_cuda"] and torch.cuda.is_available()):
    print("CUDA unavalibel! Please check the environment settings.")
    exit()

# 固定随机数种子
random.seed(args["environment"]["seed"])
np.random.seed(args["environment"]["seed"])
torch.manual_seed(args["environment"]["seed"])
torch.cuda.manual_seed_all(args["environment"]["seed"])

# cudnn网络训练加速
import torch.backends.cudnn as cudnn

# 固定加速算法
if args["environment"]["reemerge"]:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.benchmark = True

#################################### 导入模型 ###############################
from models import get_model

# 加载模型
model = get_model(args["model"]["model_name"])(args)                      # 创建模型实例(注意查看models文件夹下的__init__.py)

if args["path"]["loadmodel"] is not None:              # 加载预训练模型
    pretrain_dict = torch.load(args["path"]["loadmodel"])

    from collections import OrderedDict
    new_state_dict = OrderedDict()

    valid_state_name = ['state_dict', 'net', 'model']                             # 不同网络可能的加载字典key
    state_name = None
    for state in valid_state_name:
        if state in pretrain_dict.keys():
            state_name = state
            break

    weights = pretrain_dict if state_name is None else pretrain_dict[state_name]
    for k, v in weights.items():
        # name = k[7:] if 'module' in k and not resume else k
        name = k[7:] if 'module' in k else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)                       # 忽略部分中间输出
    # model.load_state_dict(new_state_dict)
    print('Load pretrained model.')

model = nn.DataParallel(model)              # 多GPU并行化处理模型
model.cuda()                                # 将模型搬到GPU

# 输出模型参数总量
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

###################################### 导入评价指标 ###############################
from metrics import *
metric = Metrics(args)

###################################### demo - test one case ###############################
def demo(data, save_path):

    # 使bn和dropout失效
    model.eval()

    #device = model.device
    for key, value in data.items():
        data[key] = value.cuda()

    # 设置时间节点，计算推理时间
    start_time = time.time()

    # 在不使用计算流图的情况下展开推理过程
    with torch.no_grad():
        torch.cuda.reset_max_memory_allocated()
        output = model(data)
        if 'depth' in output.keys():
            predict = output['depth']
        else:
            pt_disp = output['disp']
            focal = data['K'][:,1][:,None,None,None].type_as(data['rgb1'])
            baseline = data['K'][:,0][:,None,None,None].type_as(data['rgb1'])
            predict = (focal*baseline) / (pt_disp + 1e-8)

        # 将结果限制在(0,255): 注意，可能会缩小loss,模型输出为视差，根据给定的条件，视差最大为192
        out = torch.clamp(predict, 1, args["model"]["max_depth"])      # B × 1 × H × W，单位m

        running_time = time.time() - start_time

        print("CUDA Max Memory Occupy:", torch.cuda.max_memory_allocated() / (2**20), "MB")


    # 记录运行时间
    print('Running time = %.3fs' %(time.time() - start_time))

    # 保存预测结果
    predict, predict_scale = Tensor2PIL(out, is_percentile=False, max_depth=90)
    predict.save(save_path + '/' + 'predict.png')
    # 保存当前输出对应的左视图
    tool = Dep2PcdTool()
    imgL = data['rgb1']
    imgL = tool.renormalize(imgL)
    imgL = np.array(imgL[0].data.cpu(), dtype='uint8')
    imgL = Image.fromarray(imgL)
    imgL.save(save_path + '/' + 'input_left.png')

    # 若需要和真值图进行对比
    if 'gt_depth' in data.keys():
        disp_true = data['gt_depth']
        disp_true = disp_true.cuda()                        # B * 1 * H * W
        mask = disp_true > 1e-3
        print('Valid pixel in G.T. = ', len(disp_true[mask]))

        # 计算 L1-Loss，输出每个像素上的平均 L1
        loss_dict = {}

        metrics_result = metric(out, disp_true).data.cpu().numpy()
        # print(metrics_result)

        # # MAE(mm)
        # MAE = MAE_loss(predict, disp_true).mean()
        loss_dict['RMSE']  = metrics_result[0,0]*1000
        loss_dict['MAE']   = metrics_result[0,1]*1000
        loss_dict['iMAE']  = metrics_result[0,3]*1000
        loss_dict['iRMSE'] = metrics_result[0,2]*1000
        print('RMSE Loss = %.3f' %(loss_dict['RMSE']))

        # 保存 Error Map，便于显著比较
        error = torch.zeros_like(out)
        error = error.cuda()                                    # 放到cuda上
        error[mask] = torch.abs(out[mask] - disp_true[mask])    # 差的绝对值
        error_map, error_scale = Tensor2PIL(error, is_percentile=False,  max_depth=2)
        error_map.save(save_path + '/error-RMSE-{:.3f}.png'.format(loss_dict['RMSE']))

        # 保存真值
        disp_true, gt_scale = Tensor2PIL(disp_true, is_percentile=False, max_depth=90)
        disp_true.save(save_path + '/gt_depth.png')


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

def read_depth(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    # Image.open() 打开图像为PIL格式的 W * H矩阵
    # np.array() 将矩阵转化为一维向量，便于找到最大值
    # 虽然展成了一维，但是在内存中仍然是不连续的，这个不连续使得可以从array变回图像
    image_depth = np.array(Image.open(file_name))

    # Consider empty depth
    # 除非是空的深度图，否则深度值必然大于255
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    # 深度图单个像素为16bit，因此右移8位后，可以将数值归一化到[1,256)的浮点型区间
    image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth

def Tensor2PIL(tensor, is_percentile=False, max_depth=None):
    tensor = tensor.cpu()                       # 将tensor搬到cpu
    tensor = tensor.squeeze(0)                  # 1 * C * H * W -> C * H * W
    tensor = tensor.permute(1, 2, 0)            # H * W * C, 对应于cv2的图像处理格式
    numpy  = tensor.numpy()
    mask = numpy <= 0                           # H * W *  C
    mask = np.squeeze(mask)                     # H * W

    if max_depth is None:
        if is_percentile:
            max_depth = np.percentile(numpy, 95)    # 超过95%的深度值
        else:
            max_depth = np.max(numpy)
    alpha  = 255 / max_depth                    # 缩放因子
    
    # 将深度图转为伪彩色图像
    img = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(max_depth - numpy, 0, max_depth), alpha=alpha), cv2.COLORMAP_JET)
    print(img.shape)
    img[mask, :] = 0
    img = Image.fromarray(img)
    return img, alpha

if __name__ == '__main__':

    # 数据加载
    equip_gt = True
    datapath = "./compare_image/scene1"
    savepath = datapath + "/result/" + args['model']['model_name']
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # 输入各个图像的绝对路径(后续可设计针对多张图像的demo代码)
    path_left    = datapath + '/' + 'left.png'
    path_right   = datapath + '/' + 'right.png'
    path_Lsparse = datapath + '/' + 'Lsparse.png'
    path_Rsparse = datapath + '/' + 'Rsparse.png'
    path_dense   = datapath + '/' + 'Ldense.png'
    path_calib   = datapath + '/' + 'calib_cam_to_cam.txt'

    # 读取校准文件
    calib = read_calib_file(path_calib)                   # 获取以{key,value}为组成的字典
    P_rect_02 = np.reshape(calib['P_rect_02'], (3, 4))  # 相机02的投影矩阵
    P_rect_03 = np.reshape(calib['P_rect_03'], (3, 4))  # 相机03的投影矩阵

    # 基线计算方法参考我的论文附录2，单位：m
    baseline = P_rect_02[0,3] / P_rect_02[0,0] - P_rect_03[0,3] / P_rect_03[0,0]

    # 获取相机焦距，单位：像素
    focal = P_rect_02[0,0]

    # 测试阶段，输出基线和焦距
    print('baseline = %.2f; focal = %.2f' %(baseline, focal))

    # 读取文件
    rgb1 = Image.open(path_left)      # PIL 格式，    H * W, mode = RGB
    rgb2 = Image.open(path_right)
    dep1 = read_depth(path_Lsparse)    # np.array格式  H * W * 1, [1,256)
    dep1 = Image.fromarray(dep1.astype('float32'), mode='F')    # 转换为PIL格式，H * W, mode = F， 32位浮点型数值(无最大值限制)
    dep2 = read_depth(path_Rsparse)    # np.array格式  H * W * 1, [1,256)
    dep2 = Image.fromarray(dep2.astype('float32'), mode='F')    # 转换为PIL格式，H * W, mode = F， 32位浮点型数值(无最大值限制)

    # 网络中的上下采样操作对图像的尺寸有要求，对输入进行轻微裁剪
    w, h   = rgb1.size                # C * H * W
    w_crop = 1242#(w // 32) * 32           # 最大下采样尺寸为原图的 1/8
    h_crop = 375#256#(h // 32) * 32
    h_start = h - h_crop            # 顶部裁剪
    w_start = (w - w_crop) // 2     # 中心裁剪

    rgb1 = TF.crop(rgb1, h_start, w_start, h_crop, w_crop)
    rgb2 = TF.crop(rgb2, h_start, w_start, h_crop, w_crop)
    dep1 = TF.crop(dep1, h_start, w_start, h_crop, w_crop)
    dep2 = TF.crop(dep2, h_start, w_start, h_crop, w_crop)

    # 转为tenser格式
    rgb1 = TF.to_tensor(rgb1)      # 类型转换，不改变数值范围
    rgb1 = TF.normalize(rgb1, (0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225), inplace=True)
    # rgb1 = np.array(rgb1).astype(np.float32)
    # rgb1 = TF.normalize(TF.to_tensor(rgb1), (90.995, 96.2278, 94.3213),
    #                      (79.2382, 80.5267, 82.1483), inplace=True)
    rgb2 = TF.to_tensor(rgb2)
    rgb2 = TF.normalize(rgb2, (0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225), inplace=True)

    dep1 = TF.to_tensor(np.array(dep1))
    dep2 = TF.to_tensor(np.array(dep2))

    disp1 = baseline*focal / (dep1 + 1e-8) * (dep1 > 1e-3).float()
    disp2 = baseline*focal / (dep2 + 1e-8) * (dep2 > 1e-3).float()

    K  = torch.from_numpy(np.array([baseline, focal]).astype('float32'))
    K1 = torch.from_numpy(P_rect_02.astype('float32'))
    K1[0,2] -= w_start
    K1[1,2] -= h_start
    K2 = torch.from_numpy(P_rect_03.astype('float32'))
    K2[0,2] -= w_start
    K2[1,2] -= h_start

    # unsqueeze(0) 添加 Batch 维度
    data = {'rgb1': rgb1.unsqueeze(0), 'rgb2':rgb2.unsqueeze(0),
            'dep1': dep1.unsqueeze(0), 'dep2':dep2.unsqueeze(0),
            'disp1':disp1.unsqueeze(0), 'disp2':disp2.unsqueeze(0),
            'K1':K1.unsqueeze(0), 'K2':K2.unsqueeze(0), 'K':K.unsqueeze(0)}

    # 根据是否有gt进行相应的操作
    if equip_gt:
        dense  = read_depth(path_dense)     # np.array格式，[1,256)
        
        dense = Image.fromarray(dense.astype('float32'), mode='F')
        dense = TF.crop(dense, h_start, w_start, h_crop, w_crop)
        dense = TF.to_tensor(np.array(dense))

        data['gt_depth'] = dense.unsqueeze(0)

    demo(data, savepath)
