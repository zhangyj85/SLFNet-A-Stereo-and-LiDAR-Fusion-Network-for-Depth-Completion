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
import matplotlib.pyplot as plt

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

################################## 用于加载数据集的函数 #################################
from dataset import kittidc as DA

# 是否丢弃最后一个 batch, 多 GPU 下丢弃避免分配不均报错
drop_last = False
if len(args["train"]["gpus_id"]) > 1:
    drop_last = True

# 加载数据集
if args["environment"]["only_test"]:
    TestImgLoader = torch.utils.data.DataLoader(                            # 加载测试集
         DA.KITTIDC(args, args["test"]["split"]),
         batch_size = 1, shuffle = False, num_workers = 4, drop_last = False)
else:
    TrainImgLoader = torch.utils.data.DataLoader(
             DA.KITTIDC(args, 'train'),
             batch_size = args["train"]["batch_size"], shuffle = True,  num_workers = 8, drop_last = drop_last)
    ValImgLoader = torch.utils.data.DataLoader(                            # 加载验证集
             DA.KITTIDC(args, 'val'),
             batch_size = args["train"]["batch_size"], shuffle = False, num_workers = 8, drop_last = drop_last)

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

    # model.load_state_dict(new_state_dict, strict=False)                       # 忽略部分中间输出
    model.load_state_dict(new_state_dict)
    print('Load pretrained model.')

model = nn.DataParallel(model)              # 多GPU并行化处理模型
model.cuda()                                # 将模型搬到GPU

# 输出模型参数总量
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

###################################### 导入评价指标 ###############################
from metrics import *
metric = Metrics(args)


###################################### 模型训练 ##################################
def batch_train(imgL, imgR, imgLS, imgRS, gt_depth, calib):

        model.train()           # 使 BN 和 dropout 生效

        # 将数据放到GPU
        imgL, imgR, imgLS, imgRS, gt_depth, calib = imgL.cuda(), imgR.cuda(), imgLS.cuda(), imgRS.cuda(), gt_depth.cuda(), calib.cuda()

        # 梯度清零
        optimizer.zero_grad()

        # SLFnet 模型输出
        PAM_disp, Mask_disp, SPN_inter, pt_disp,\
        PAM_mask, V_left_to_right, V_right_to_left,\
        M_left_to_right, M_right_to_left,\
        M_left_right_left, M_right_left_right = model(imgL, imgR, imgLS, imgRS, calib, is_training=True)
        
        ### loss
        loss = loss_function((PAM_disp, Mask_disp, SPN_inter, pt_disp), PAM_mask,
                             gt_depth, calib,
                             [(M_right_to_left, M_left_to_right),
                              (M_left_right_left, M_right_left_right),
                              (V_left_to_right, V_right_to_left)],
                             (imgL, imgR), vis=False)

        loss_dict = loss_function.loss_dict

        loss.backward()         # 反向传播
        optimizer.step()        # 优化器学习率更新

        return loss.data.cpu(), loss_dict  # 据说一定要这样，因为有坑

def model_train():

    # 加载进度条, 可视化训练进度
    from tqdm import tqdm

    # 加载损失函数
    loss_function = DesignLoss(args)            # 创建损失函数实例
    loss_function.cuda()

    # 选择优化器Adam
    optimizer = optim.Adam([paras for paras in model.parameters() if paras.requires_grad == True],
                           lr=args["train"]["lr"],
                           betas=(0.9, 0.999))

    # 确保训练结果保存路径是存在的
    if not os.path.exists(args["path"]["savemodel"]):
        os.mkdir(args["path"]["savemodel"])

    ### train ###
    start_full_time = time.time()               # 记录总训练时长

    if args["path"]["loadmodel"] is not None:
        # 继续断点训练
        s = args["path"]["loadmodel"][-6:-4]               # 取.tar前的2个字符
        if s[0] == '_':
            s = s[1]
        checkpoint_num = int(s, 10) + 1
        print('From checkpoint %d continue the training.' %(checkpoint_num))
    else:
        checkpoint_num = 0

    best_epoch = -1
    best_loss = float('inf')
    for epoch in range(checkpoint_num, args["train"]["max_epochs"]):
        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        loss_list = []
        adjust_learning_rate(optimizer, epoch)  # 根据epoch调整学习率

        ## epoch training ##
        bar = tqdm(total=args["train"]["report_batch"])
        for batch_idx, data in enumerate(TrainImgLoader):

            imgL_crop  = data['left']            # size = B * C * H * W; H*W=256*1024
            imgR_crop  = data['right']
            imgLS_crop = data['L_sparse']
            imgRS_crop = data['R_sparse']
            gt_L_crop  = data['dense']
            K          = data['K']               # 包括基线(m)和焦距(pixel)

            start_time = time.time()            # 每个batch计时一次
            loss, loss_dict = train(imgL_crop, imgR_crop, imgLS_crop, imgRS_crop, gt_L_crop, K)

            total_train_loss += loss            # 累计loss
            loss_list.append(loss_dict)         # 记录单个epoch内的训练loss变化

            # 每 20 个 batch 输出训练信息, 最后一个也输出
            if (batch_idx % args["train"]["report_batch"] == 0) or (batch_idx == len(TrainImgLoader) - 1):
                # 暂时保存参数
                savefilename = args["path"]["savemodel"] + '/batch_save.tar'
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss / len(TrainImgLoader),
                    'loss_list': loss_list,
                }, savefilename)

                bar.reset()
                bar.write(" ")     # 换行
                bar.write('Epoch:%d, Iter:%d | Total:%d. Training loss = %.3f , time = %.2f' %(epoch, batch_idx, len(TrainImgLoader)-1, loss, time.time() - start_time))
                # 打印 loss 细节
                for key in loss_dict:
                    if isinstance(loss_dict[key], tuple):
                        bar.write(key + ':\t' + \
                        'smooth : ' + '%.2f\t' %(loss_dict[key][0].item()) + \
                        'cycle  : ' + '%.2f\t' %(loss_dict[key][1].item()) + \
                        'photo  : ' + '%.2f\t' %(loss_dict[key][2].item()) )
                    else:
                        bar.write(key + ':' + '%.2f' %(loss_dict[key].item()))
            bar.update(1)
        bar.close()

        # 完成一个 epoch, 输出汇总的 loss
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))

        # 完成一个epoch的训练，保存参数
        savefilename = args["path"]["savemodel"]+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
            'loss_list': loss_list,
        }, savefilename)

        ## epoch test ##
        total_test_loss = 0     # 总loss
        loss_list = []          # 每个样本的loss
        time_list = []
        RMSE_list = []

        for batch_idx, data in enumerate(ValImgLoader):

            imgL   = data['left']
            imgR   = data['right']
            imgLS  = data['L_sparse']
            imgRS  = data['R_sparse']
            disp_L = data['dense']
            K      = data['K']               # 包括基线(m)和焦距(pixel)

            start_time = time.time()
            test_loss, running_time = test(imgL, imgR, imgLS, imgRS, disp_L, K)
            print('Iter:%d | Total:%d. Test loss = %.3f , time = %.2f' %(batch_idx, len(ValImgLoader)-1, test_loss['RMSE'], time.time() - start_time))

            loss_list.append(test_loss)
            RMSE_list.append(test_loss['RMSE'])
            time_list.append(running_time)

        # Save test information
        savefilename = args["path"]["savemodel"] + '/val_information_'+str(epoch)+'.tar'
        torch.save({
            'loss_list': loss_list,
            'time_list': time_list,
            }, savefilename)

        # 记录最好的成绩
        if best_loss > sum(RMSE_list) / len(RMSE_list):
            best_loss = sum(RMSE_list) / len(RMSE_list)
            best_epoch = epoch

            savefilename = args["path"]["savemodel"] + '/TheBest.tar'
            torch.save({
                'best_epoch': best_epoch,
                'best_loss': best_loss,
                'state_dict': model.state_dict(),
                }, savefilename)

    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))


def adjust_learning_rate(optimizer, epoch):         # 修改Adam学习率
    lr = args["train"]["lr"]
    if epoch > 3:
        lr = args["train"]["lr"] * 0.5
    if epoch > 8:
        lr = args["train"]["lr"] * 0.1
    if epoch > 14:
        lr = args["train"]["lr"] * 0.01
    print("In %d-th epoch, the learning rate is : %.6f" %(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

########################################### 模型测试 ############################################
def batch_test(data):

    # 使bn和dropout失效
    model.eval()

    # 设置时间节点，计算推理时间
    start_time = time.time()

    # 在不使用计算流图的情况下展开推理过程
    with torch.no_grad():
        output = model(data)

        # 网络并非直接输出深度, 则将视差转为深度
        if "depth" not in output.keys():
            calib = data['K']
            focal = calib[:,1][:,None,None,None].type_as(data['rgb1'])
            baseline = calib[:,0][:,None,None,None].type_as(data['rgb1'])
            predict = (focal * baseline) / (output['disp'] + 1e-8)
        else:
            predict = output['depth']

        # 将结果限制在(1,100)
        predict = torch.clamp(predict, 1, args["model"]["max_depth"])      # B × 1 × H × W，单位 m

    # 记录运行时间
    running_time = time.time() - start_time

    # 多个loss计算
    loss_dict = {}

    metrics_result = metric(predict, data['gt_depth']).data.cpu().numpy()

    loss_dict['RMSE']  = metrics_result[0,0]*1000
    loss_dict['MAE']   = metrics_result[0,1]*1000
    loss_dict['iMAE']  = metrics_result[0,3]*1000
    loss_dict['iRMSE'] = metrics_result[0,2]*1000

    return loss_dict, running_time

def model_test():

    # 确保测试结果保存路径是存在的
    savepath = os.path.join(args["path"]["savemodel"], args["model"]["model_name"], 'test')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    total_test_loss = 0     # 总loss
    loss_list = []          # 每个样本的loss
    time_list = []

    for batch_idx, data in enumerate(TestImgLoader):

        # 数据载入 cuda
        for key, value in data.items():
            data[key] = value.cuda()

        start_time = time.time()
        test_loss, running_time = batch_test(data)
        print('Iter:%d | Total:%d. Test loss = %.3f , time = %.2f' %(batch_idx + 1, len(TestImgLoader), test_loss['RMSE'], running_time))

        loss_list.append(test_loss)
        time_list.append(running_time)

    # Save test information
    savefilename = os.path.join(savepath, 'model_test.tar')
    torch.save({
        'loss_list': loss_list,
        'time_list': time_list,
        }, savefilename)


if __name__ == '__main__':

    if args["environment"]["only_test"]:
        model_test()

    else:

        if args["model"]["model_name"].lower() == "slfnet":
            model_train()
        else:
            raise ValueError(
            f"Model {name} not support training. Valid model is SLFnet."
        )