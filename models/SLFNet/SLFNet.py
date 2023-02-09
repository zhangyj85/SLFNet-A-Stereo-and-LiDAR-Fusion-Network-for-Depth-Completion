from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
from .basicblock import *

class SLFNet(nn.Module):
    def __init__(self, args):
        super(SLFNet, self).__init__()                          # 继承父类nn.Module的__init__属性
        '''
        1/1     1/2     1/4     1/8     1/16      1/32
        16      32      64      96      128       160
        '''

        self.max_disp = args["model"]["max_disp"]
        channels = [16, 32, 64, 96, 128, 160]

        # PAMstereo block: using stereo information to get an init depth map / disp map
        # self.layer0_rgb = conv_bn_relu(3, 16, 3, 1, 1)              # 图像预编码
        # self.layer0_sparse = conv_bn_relu(1, 8, 3, 1, 1)            # sparse depth 预编码
        # self.layer0_dense = conv_bn_relu(1, 8, 3, 1, 1)             # dense depth 编码
        self.rgb_encoder    = encoder(inplanes=16, channels=channels)
        self.sparse_encoder = encoder(inplanes=32, channels=channels)
        self.trans_rgb = conv_bn_relu(3, 16, kernel_size=1, stride=1, padding=0, bn=True, relu=True)
        self.trans_dep = conv_bn_relu(1, 16, kernel_size=1, stride=1, padding=0, bn=True, relu=True)
        self.PAM = PAM(channels=channels[2], args=args)

        # Mask depth block: 利用conv实现sparse&stereo的深度互补及各项同性传播
        self.dense_encoder = encoder(inplanes=1, channels=channels)
        self.refine1 = decoder(inplanes=channels[2]*3, outplanes=2, channels=channels)
        self.conf_sigmoid = nn.Sigmoid()
        self.offset_lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Invalid depth block: 利用NLSPN进行遮挡区域的深度恢复
        self.get_guidence0 = conv_bn_relu(channels[0]*2, 32, 3, 1, 1)
        self.get_guidence1 = conv_bn_relu(32+channels[0]*2, 32, 3, 1, 1)
        self.refine2 = NLSPN(args, ch_g=32, ch_f=1, k_g=3, k_f=3)

        self.pool = nn.MaxPool2d((2,2), 2)

        for m in self.modules():                                    # 网络模型权重初始化
            if isinstance(m, nn.Conv2d):                            # 对Conv2d的正态分布初始化
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.xavier_normal_(m.weight.data)                # 对Conv2d的初始化方法参考了PAM
            elif isinstance(m, nn.BatchNorm2d):                     # 对BN的初始化
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):                          # 对全连接层的初始化(useless)
                m.bias.data.zero_()


    def forward(self, data, *args):
        imgL = data['rgb1']
        imgR = data['rgb2']
        sparseL = data['disp1']
        sparseR = data['disp2']

        # 最大下采样 1/32
        if not self.training:
            # 最大下采样 32, 需要根据输入进行相应的 padding
            top_pad  = math.ceil(imgL.shape[-2] / 32) * 32 - imgL.shape[-2]
            left_pad = math.ceil(imgL.shape[-1] / 32) * 32 - imgL.shape[-1]
            imgL = F.pad(imgL, (left_pad, 0, top_pad, 0), 'constant', 0)
            imgR = F.pad(imgR, (left_pad, 0, top_pad, 0), 'constant', 0)
            sparseL = F.pad(sparseL, (left_pad, 0, top_pad, 0), 'constant', 0)
            sparseR = F.pad(sparseR, (left_pad, 0, top_pad, 0), 'constant', 0)

            # 试一下 resize 的效果
            # _,_,oh,ow = imgL.shape
            # imgL = F.interpolate(imgL, (256, 1024), mode='bilinear')
            # imgR = F.interpolate(imgR, (256, 1024), mode='bilinear')
            # sparseL = self.pool(F.interpolate(sparseL, (512, 2048), mode='nearest')) * 1024/ow
            # sparseR = self.pool(F.interpolate(sparseR, (512, 2048), mode='nearest')) * 1024/ow

        # 输入初始化
        sl_mask = (sparseL > 1).float()
        sr_mask = (sparseR > 1).float()
        imgL_init = self.trans_rgb(imgL)
        imgR_init = self.trans_rgb(imgR)
        sparseL_init = self.trans_dep(sparseL / self.max_disp)
        sparseR_init = self.trans_dep(sparseR / self.max_disp)

        sparseL_init = torch.cat((imgL_init, sparseL_init), dim=1) * sl_mask
        sparseR_init = torch.cat((imgR_init, sparseR_init), dim=1) * sr_mask

        # 对所有输入图像进行编码
        imgL_fea_out, imgL_fea_14, imgL_fea_12, imgL_fea_11 = self.rgb_encoder(imgL_init)
        imgR_fea_out, imgR_fea_14, imgR_fea_12, imgR_fea_11 = self.rgb_encoder(imgR_init)
        dispL_fea_out, dispL_fea_14, dispL_fea_12, dispL_fea_11 = self.sparse_encoder(sparseL_init)    # 数值归一化, 192为kitti最大视差
        dispR_fea_out, dispR_fea_14, dispR_fea_12, dispR_fea_11 = self.sparse_encoder(sparseR_init)    # 数值归一化, 192为kitti最大视差

        left_feature  = imgL_fea_out + dispL_fea_out
        right_feature = imgR_fea_out + dispR_fea_out

        # PAMstereo block:
        # rgb + sparse 基于这样一种假设, 即点云在左图出现的地方, 在右图的非遮挡区域同样存在, 相加可以增强对应区域的一致性, 相对地削弱了无纹理区域、重复纹理区域的错误匹配可能
        PAM_disp, M_right_to_left, M_left_to_right, \
        M_left_right_left, M_right_left_right, \
        V_left_to_right, V_right_to_left = self.PAM(left_feature, right_feature)

        # 视差图上采样, 计算初始预测深度
        PAM_disp = F.interpolate(PAM_disp, (imgL.size()[2], imgL.size()[3]), mode='bilinear', align_corners=True)
        PAM_disp = PAM_disp * 4                               # PAM_disparity原本的视差是在1/4, 因此*4补偿回原尺寸

        # 视差mask上采样, 该方法有待商榷
        PAM_mask = F.interpolate(V_left_to_right, (imgL.size()[2], imgL.size()[3]), mode='bilinear', align_corners=True)
        PAM_mask = (PAM_mask > 0.5).float()

        # mask block:
        # sparse_mask = (L_sparse > 1e-4).float()
        # internal_disp = (PAM_disp + L_sparse) / (1.0 + sparse_mask)     # 同时有值, 取平均; 否则取单个值. PAM_disp始终有值
        # L_dense_f0 = self.layer0_dense(internal_disp / 192)             # 同理, 进行数值归一化操作

        # 既然输入的sparse有误差, 那么不直接相加, 由网络自己决定学习策略
        fusion_feature, fusion_14, fusion_12, fusion_11 = self.dense_encoder(PAM_disp / self.max_disp)

        B, C, H, W = right_feature.shape
        right_feature = torch.bmm(M_right_to_left.contiguous().view(-1, W, W), 
                                  right_feature.permute(0,2,3,1).contiguous().view(-1, W, C)
                                  ).contiguous().view(B, H, W, C).permute(0,3,1,2)  #  B * C * H * W
        imgR_fea_14   = torch.bmm(M_right_to_left.contiguous().view(-1, W, W),
                                  imgR_fea_14.permute(0,2,3,1).contiguous().view(-1, W, C)
                                  ).contiguous().view(B, H, W, C).permute(0,3,1,2)  #  B * C * H * W

        decoder_out, decoder_11, decoder_12, decoder_14 = self.refine1(torch.cat((left_feature, right_feature, fusion_feature), dim=1),
                                                                       torch.cat((imgL_fea_14, imgR_fea_14, fusion_14, dispL_fea_14), dim=1),
                                                                       torch.cat((imgL_fea_12, fusion_12, dispL_fea_12), dim=1),
                                                                       torch.cat((imgL_fea_11, fusion_11, dispL_fea_11), dim=1))
        Mask_offset = self.offset_lrelu(decoder_out[:, 0, :, :].view(B, 1, 4*H, 4*W))
        Mask_conf   = self.conf_sigmoid(decoder_out[:, 1, :, :].view(B, 1, 4*H, 4*W))
        Mask_disp   = PAM_disp + Mask_offset

        # Invalid block:
        guidence = self.get_guidence0(torch.cat((fusion_11, decoder_11), dim=1))
        guidence = self.get_guidence1(torch.cat((guidence, imgL_fea_11, dispL_fea_11), dim=1))
        pt_disp, SPN_inter, offset, aff, aff_const = self.refine2(Mask_disp.contiguous(), guidence, Mask_conf, sparseL, imgL)

        if self.training:
            return PAM_disp, Mask_disp, SPN_inter, pt_disp,\
                   PAM_mask, V_left_to_right, V_right_to_left,\
                   M_left_to_right, M_right_to_left,\
                   M_left_right_left, M_right_left_right
        else:
            output = {"disp": torch.clamp(pt_disp, 1, self.max_disp)[:,:,top_pad:,left_pad:]}
            # output = F.interpolate(pt_disp, (oh, ow), mode='bilinear') * ow/1024
            return output
