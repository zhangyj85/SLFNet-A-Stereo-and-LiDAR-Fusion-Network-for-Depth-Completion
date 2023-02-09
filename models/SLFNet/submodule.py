'''
包含了网络的一些基本模块
'''

from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from .basicblock import *

# 用于NLSPN
from .modulated_deform_conv_func import ModulatedDeformConvFunction

# 用于PAM
from skimage import morphology

class decoder(nn.Module):
    # 关于decoder，什么时候使用res，什么时候使用conv；以及什么时候需要ReLU，什么时候conv的bias设置为True，需要重新认识
    # decoder 的下一步设计: 输出改为预测深度的bias, sparse 的置信度, PAM depth 的置信度
    def __init__(self, inplanes=64, outplanes=2, channels=[]):
        super(decoder, self).__init__()

        # 1/4，信息融合
        self.E2 = self._make_blocks(1, inplanes,    channels[2], downsample=False)              # scale: 1/4
        self.E3 = self._make_blocks(1, channels[2], channels[3], downsample=True)               # scale: 1/8
        self.E4 = self._make_blocks(1, channels[3], channels[4], downsample=True)               # scale: 1/16
        self.E5 = self._make_blocks(1, channels[4], channels[5], downsample=True)               # scale: 1/32

        # 1/16, 恢复图像结构
        self.upsample4 = convt_bn_relu(channels[5],  channels[4], kernel_size=3, stride=2, padding=1, output_padding=1, bn=True, relu=True)
        self.conv4     = conv_bn_relu(channels[4]*2, channels[4], kernel_size=3, stride=1, padding=1, dilation=1, bn=True, relu=True)

        # 1/8, 
        self.upsample3 = convt_bn_relu(channels[4],  channels[3], kernel_size=3, stride=2, padding=1, output_padding=1, bn=True, relu=True)
        self.conv3     = conv_bn_relu(channels[3]*2, channels[3], kernel_size=3, stride=1, padding=1, dilation=1, bn=True, relu=True)

        # 1/4，恢复图像结构
        self.upsample2 = convt_bn_relu(channels[3],  channels[2], kernel_size=3, stride=2, padding=1, output_padding=1, bn=True, relu=True)
        self.conv2     = conv_bn_relu(channels[2]*6, channels[2], kernel_size=3, stride=1, padding=1, dilation=1, bn=True, relu=True)

        # 1/2
        self.upsample1 = convt_bn_relu(channels[2],  channels[1], kernel_size=3, stride=2, padding=1, output_padding=1, bn=True, relu=True)
        self.conv1     = conv_bn_relu(channels[1]*4, channels[1], kernel_size=3, stride=1, padding=1, dilation=1, bn=True, relu=True)
        
        # 1/1，增加细节信息，恢复纹理结构
        self.upsample0 = convt_bn_relu(channels[1],  channels[0], kernel_size=3, stride=2, padding=1, output_padding=1, bn=True, relu=True)
        self.conv0     = conv_bn_relu(channels[0]*4, channels[0], kernel_size=3, stride=1, padding=1, dilation=1, bn=True, relu=True)

        self.finalconv = conv_bn_relu(channels[0], outplanes, kernel_size=1, stride=1, padding=0, bn=False, relu=False)

    def _make_blocks(self, n_blocks, channels_in, channels_out, downsample=False):
        body = []
        if downsample:
            downsample_layer = conv_bn_relu(channels_in, channels_out, kernel_size=1, stride=2, padding=0, dilation=1, bn=True, relu=False)
            body.append(ResBlock(channels_in, channels_out, stride=2, downsample=downsample_layer, padding=1, dilation=1))
        elif channels_in != channels_out:
            downsample_layer = conv_bn_relu(channels_in, channels_out, kernel_size=1, stride=1, padding=0, dilation=1, bn=True, relu=False)
            body.append(ResBlock(channels_in, channels_out, stride=1, downsample=downsample_layer, padding=1, dilation=1))
        else:
            body.append(ResBlock(channels_in, channels_out, stride=1, downsample=None, padding=1, dilation=1))

        for i in range(n_blocks):
            body.append(ResBlock(channels_out, channels_out, stride=1, downsample=None, padding=1, dilation=1))
        return nn.Sequential(*body)

    def forward(self, x, in_2, in_1, in_0):
        # encoder block
        fea_E2 = self.E2(x)         # 1/4
        fea_E3 = self.E3(fea_E2)    # 1/8
        fea_E4 = self.E4(fea_E3)    # 1/16
        fea_E5 = self.E5(fea_E4)    # 1/32

        # 1/16, 128
        fea_D4 = torch.cat((self.upsample4(fea_E5), fea_E4), dim=1).contiguous()
        fea_D4 = self.conv4(fea_D4)

        # 1/8,  96
        fea_D3 = torch.cat((self.upsample3(fea_D4), fea_E3), dim=1).contiguous()
        fea_D3 = self.conv3(fea_D3)

        # 1/4,  64
        fea_D2 = torch.cat((self.upsample2(fea_D3), fea_E2, in_2), dim=1).contiguous()
        fea_D2 = self.conv2(fea_D2)

        # 1/2,  32
        fea_D1 = torch.cat((self.upsample1(fea_D2), in_1), dim=1).contiguous()
        fea_D1 = self.conv1(fea_D1)

        # 1/1,  16
        fea_D0 = torch.cat((self.upsample0(fea_D1), in_0), dim=1).contiguous()
        fea_D0 = self.conv0(fea_D0)

        out = self.finalconv(fea_D0)

        return out, fea_D0, fea_D1, fea_D2


class encoder(nn.Module):
    def __init__(self, inplanes=3, channels=[]):
        super(encoder, self).__init__()
        # inplanes: rgb=3, sparse=1
        # channels: a list
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 通过 3 * 3 的卷积核提取局部特征信息, 等效 7*7
        self.E0 = nn.Sequential(conv_bn_relu(inplanes,    channels[0], kernel_size=3, stride=1, padding=1, dilation=1, bn=True, relu=True),
                                conv_bn_relu(channels[0], channels[0], 3, 1, 1, 1, bn=True, relu=True),
                                conv_bn_relu(channels[0], channels[0], 3, 1, 1, 1, bn=True, relu=True))

        self.E1 = self._make_blocks(4, channels[0], channels[1], downsample=True)               # scale: 1/2
        self.E2 = self._make_blocks(12, channels[1], channels[2], downsample=True)               # scale: 1/4
        self.E3 = self._make_blocks(1, channels[2], channels[3], downsample=True)               # scale: 1/8
        # self.E4 = self._make_blocks(1, channels[3], channels[4], downsample=True)               # scale: 1/16
        # self.E5 = self._make_blocks(1, channels[4], channels[5], downsample=True)               # scale: 1/32

        # self.D5 = self._make_blocks(1, channels[5], channels[5], downsample=False)              # scale: 1/32
        # self.D4 = self._make_blocks(1, channels[5] + channels[4], channels[4])                  # scale: 1/16
        # self.D3 = self._make_blocks(1, channels[4] + channels[3], channels[3])                  # scale: 1/8
        self.D2 = self._make_blocks(3, channels[3] + channels[2], channels[2])                  # scale: 1/4

    def _make_blocks(self, n_blocks, channels_in, channels_out, downsample=False):
        body = []
        if downsample:
            downsample_layer = conv_bn_relu(channels_in, channels_out, kernel_size=1, stride=2, padding=0, dilation=1, bn=True, relu=False)
            body.append(ResBlock(channels_in, channels_out, stride=2, downsample=downsample_layer, padding=1, dilation=1))
        elif channels_in != channels_out:
            downsample_layer = conv_bn_relu(channels_in, channels_out, kernel_size=1, stride=1, padding=0, dilation=1, bn=True, relu=False)
            body.append(ResBlock(channels_in, channels_out, stride=1, downsample=downsample_layer, padding=1, dilation=1))
        else:
            body.append(ResBlock(channels_in, channels_out, stride=1, downsample=None, padding=1, dilation=1))

        for i in range(n_blocks):
            body.append(ResBlock(channels_out, channels_out, stride=1, downsample=None, padding=1, dilation=1))
        return nn.Sequential(*body)

    def forward(self, x):
        fea_E0 = self.E0(x)                                                                 # b*16*(h/1)*(w/1)
        fea_E1 = self.E1(fea_E0)                                                            # b*32*(h/2)*(w/2)
        fea_E2 = self.E2(fea_E1)                                                            # b*64*(h/4)*(w/4)
        fea_E3 = self.E3(fea_E2)                                                            # b*96*(h/8)*(w/8)
        # fea_E4 = self.E4(fea_E3)                                                            # b*128*(h/16)*(w/16)
        # fea_E5 = self.E5(fea_E4)                                                            # b*160*(h/32)*(w/32)

        # fea_D5 = self.D5(fea_E5)                                                            # b*160*(h/32)*(w/32)
        # fea_D4 = self.D4(torch.cat((self.upsample(fea_D5), fea_E4), dim=1))                 # b*128*(h/16)*(w/16)
        # fea_D3 = self.D3(torch.cat((self.upsample(fea_D4), fea_E3), dim=1))                 # b*96*(h/8)*(w/8)
        # fea_D2 = self.D2(torch.cat((self.upsample(fea_D3), fea_E2), dim=1))                 # b*64*(h/4)*(w/4)
        fea_D2 = self.D2(torch.cat((self.upsample(fea_E3), fea_E2), dim=1))                 # b*64*(h/4)*(w/4)

        return fea_D2, fea_E2, fea_E1, fea_E0


class PAM(nn.Module):
    # PAM: 不改变输入的size, channels，仅引入双目匹配信息(视差 + 遮挡信息)
    def __init__(self, channels=64, args=None):
        super(PAM, self).__init__()

        self.max_disp = args["model"]["max_disp"] // 4
        if self.max_disp <= 0:
            self.max_disp = 192//4

        self.rb = ResBlock(channels, channels, stride=1, downsample=None, padding=1, dilation=1)

        # 1*1 Conv
        # 在实现 attention 的过程中不采用 bn 和 relu, 从而保证 QKV 得到的数值有较大的动态范围, 而不是限制在 (0,1) 区间内
        # 从而保证进行矩阵乘法后得到的 attention map 是一个稀疏张量, 满足预期 (否则为一个稠密张量, 没有实现attention的能力)
        self.b1 = nn.Sequential(conv_bn_relu(channels, channels, 3, 1, 1, bn=False, relu=True),
                                conv_bn_relu(channels, channels, 1, 1, 0, bn=False, relu=False))    # 注意这里引入了偏置, 理解为输入与输出之间的映射偏置
        self.b2 = nn.Sequential(conv_bn_relu(channels, channels, 3, 1, 1, bn=False, relu=True),
                                conv_bn_relu(channels, channels, 1, 1, 0, bn=False, relu=False))

    def forward(self, x_left, x_right):
        b, c, h, w   = x_left.shape
        buffer_left  = self.rb(x_left)  # 将图像送入残差块
        buffer_right = self.rb(x_right)

        if self.max_disp > w:
            self.max_disp = w
        Invalid_L2R = Variable(torch.tril(torch.ones(b*h, w, w), diagonal=-1), requires_grad=False).cuda().detach() * (-1e8) + \
                      Variable(torch.triu(torch.ones(b*h, w, w), diagonal=+self.max_disp), requires_grad=False).cuda().detach() * (-1e8)
        Invalid_R2L = Variable(torch.triu(torch.ones(b*h, w, w), diagonal=+1), requires_grad=False).cuda().detach() * (-1e8) + \
                      Variable(torch.tril(torch.ones(b*h, w, w), diagonal=-self.max_disp), requires_grad=False).cuda().detach() * (-1e8)

        ### M_{right_to_left}, (W_left * W_right), 下三角有效
        Q = self.b1(buffer_left).permute(0, 2, 3, 1)                                                # B * H * W * C
        S = self.b2(buffer_right).permute(0, 2, 1, 3)                                               # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w)) / c                                        # (B*H) * W * W, /sqrt(c) 对尺度进行归一化, 使输入和输出的方差一致, 理由看 Attention is all you need
        # 编写 softmax 函数, 以保证 取上/下三角的过程始终是正确的
        # score = score * DownTril                                                                  # 将负视差(上三角区域)清零
        score = score + Invalid_R2L                                                                 # 将负视差区域的数值置为负无穷
        score = torch.exp(score - score.max(-1)[0].unsqueeze(-1))                                   # softmax 的分子实现
        # score = score * DownTril                                                                    # 经过 softmax 之后, 原本清零的负视差不为零, 再次清零
        M_right_to_left = score / (score.sum(-1, keepdim=True) + 1e-8)                              # 权重归一化, 得到理想的 softmax 输出

        ### M_{left_to_right}, (W_right * W_left), 上三角有效
        Q = self.b1(buffer_right).permute(0, 2, 3, 1)                                               # B * H * W * C
        S = self.b2(buffer_left).permute(0, 2, 1, 3)                                                # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w)) / c                                        # (B*H) * W * W
        # 编写 softmax 函数, 以保证 取上/下三角的过程始终是正确的
        # score = score * UpTril                                                                    # 将负视差清零
        score = score + Invalid_L2R                                                                 # 将负视差区域的数值置为负无穷
        score = torch.exp(score - score.max(-1)[0].unsqueeze(-1))                                   # softmax 的分子实现
        # score = score * UpTril                                                                      # 经过 softmax 之后, 原本清零的负视差不为零, 再次清零
        M_left_to_right = score / (score.sum(-1, keepdim=True) + 1e-8)                              # 权重归一化, 得到理想的 softmax 输出

        # 测试一下得到的Ｍ矩阵是否已经为上三角
        # print("M sum:", torch.tril(M_left_to_right, diagonal=-1).reshape(-1).sum())
        # print("score:", M_left_to_right.sum(-1).reshape(-1).sum())

        ### valid masks
        # 有效定义为能够利用左图恢复出右图。若M_left_to_right中的某个像素位置的加权<0.1，表示左图中的该像素在右图没有对应的像素
        # V_left_to_right * left, 得到的 left 图的每个像素都能在右图找到对应的像素
        # 对应于遮挡。即无效区域。反之则为有效。0.1为超参数
        V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1                              #  BH * W_right * W_left, 沿 列方向 求和>1为之有效 
        V_left_to_right = V_left_to_right.view(b, 1, h, w)                                          #  B * 1 * H * W
        V_left_to_right = morphologic_process(V_left_to_right)                                      #  形态学处理

        if self.training:
            V_right_to_left = torch.sum(M_right_to_left.detach(), 1) > 0.1
            V_right_to_left = V_right_to_left.view(b, 1, h, w)                                      #  B * 1 * H * W
            V_right_to_left = morphologic_process(V_right_to_left)

            M_left_right_left = torch.bmm(M_right_to_left, M_left_to_right)
            M_right_left_right = torch.bmm(M_left_to_right, M_right_to_left)

        ### position, 图像位置编码, 用以输出视差图, 注意repeat的用法是 (a, b) -> repaet(c, d, e, f) -> (c, d, e*a, f*b)
        # 为了统一整个流程, 从右视图生成左视图
        right_position = Variable(torch.FloatTensor([i for i in range(0, w)]).repeat(b, 1, h, 1), requires_grad=False).cuda().detach()   # b1hw
        left_position = torch.bmm(M_right_to_left.contiguous().view(b*h, w, w), right_position.permute(0, 2, 3, 1).contiguous().view(b*h, w, 1))
        left_position = left_position.view(b, h, w, 1).contiguous().permute(0, 3, 1, 2)
        PAM_disparity = right_position - left_position                                              # 获得的视差图的取值范围在 0-W 注意不同尺度下的W应当进行缩放, 是视差的意义对齐
        # 注意, 这里得到的视差图还包括了从遮挡区域获取的错误视差, 需要进行mask和相应的插值处理
        PAM_disparity = disparity_fill_hold(PAM_disparity, V_left_to_right)

        # check whether PAM Disparity is already right
        # print("PAM Disparity > 0:", (PAM_disparity > 0).float().sum())
        # print("PAM Disparity < 0:", (PAM_disparity < 0).float().sum())
        # print("PAM xxx : \n", PAM_disparity[PAM_disparity < 0])

        ### fusion
        # buffer = self.b3(x_right).permute(0,2,3,1).contiguous().view(-1, w, c)                      # (B*H) * W *  C
        # buffer = torch.bmm(M_right_to_left, buffer).contiguous().view(b, h, w, c).permute(0,3,1,2)  #  B * C * H * W

        # out = self.fusion(torch.cat((V_left_to_right,                                               # B * 1 * H * W
        #                              x_left,                                                        # B * C * H * W
        #                              buffer,                                                        # B * C * H * W
        #                              PAM_disparity.contiguous().view(b, 1, h, w)), dim=1))          # B * 1 * H * W

        ## output
        # 这里标注一下, 方便之后改进
        # 由于没有限制视差的最大值, 192的视差值甚至有可能被预测为 W(1024), 虽然相对于深度而言仅仅是不到1m的预测误差, 但并没有很好的利用双目视差的约束
        # 后续可以采用视差体和深度体的共同优势获取更精确的深度信息
        if self.training:
            return PAM_disparity, M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w), \
                   M_left_right_left.view(b,h,w,w), M_right_left_right.view(b,h,w,w), \
                   V_left_to_right, V_right_to_left
        else:
            return PAM_disparity, M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w), \
                   None, None, \
                   V_left_to_right, None

def morphologic_process(mask):
    device  = mask.device
    b,_,_,_ = mask.shape
    mask    = ~mask                                                      # 此时 mask 为 1 的地方表示无效区域
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)               # 移除 mask=1 且 8连通区域面积小于20的 无效区域
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)                 # 移除 mask=0 且 4连通区域面积小于10的 有效面积
    for idx in range(b):
        buffer = np.pad(mask_np[idx,0,:,:],((3,3),(3,3)),'constant')        # 对mask_np在 h * w 平面上下、左右各 padding 3行0
        buffer = morphology.binary_closing(buffer, morphology.disk(3))      # 对 buffer 进行闭运算，kernel = disk(半径为3的圆)
        mask_np[idx,0,:,:] = buffer[3:-3,3:-3]                              # 取图像本来的大小
    mask_np = 1-mask_np
    mask_np = mask_np.astype(float)

    return torch.from_numpy(mask_np).float().to(device)                     # 将 mask 转为浮点型 tensor 并放回device中

def disparity_fill_hold(disp_init, valid_mask):
    # 构建(1,1,0)滤波算子
    filter1 = torch.zeros(1,3).to(disp_init.device)
    filter1[0, 0] = 1
    filter1[0, 1] = 1
    filter1 = filter1.view(1,1,1,3)

    # 构建(0,1,1)滤波算子
    filter2 = torch.zeros(1,3).to(disp_init.device)
    filter2[0, 1] = 1
    filter2[0, 2] = 1
    filter2 = filter2.view(1,1,1,3)

    # 迭代
    valid_mask_0 = valid_mask
    disp = disp_init * valid_mask_0

    # 输入的 disp_init 是从右图生成的, 因此生成的左图平面坐标下的视差图在图像的左侧一定存在遮挡区域, 所以应该从右边开始填充?
    # 此外, 从信息的角度来讲, 生成左图中的元素中, 越靠近右侧, 则能够利用越多的右图信息, 因此获取的视差值也越准确
    # 利用右边的像素进行填充
    valid_mask_num = 1
    while valid_mask_num > 0:
        valid_mask_1 = F.conv2d(valid_mask_0, filter1, padding=[0,1])
        disp = disp * valid_mask_0 + \
               F.conv2d(disp, filter1, padding=[0,1]) / (valid_mask_1 + 1e-4) * ((valid_mask_1 > 0).float() - valid_mask_0)
        valid_mask_num = (valid_mask_1 > 0).float().sum() - valid_mask_0.sum()
        valid_mask_0 = (valid_mask_1 > 0).float()

    # 利用左边的像素进行填充
    valid_mask_num = 1
    while valid_mask_num > 0:
        valid_mask_1 = F.conv2d(valid_mask_0, filter2, padding=[0,1])
        disp = disp * valid_mask_0 + \
               F.conv2d(disp, filter2, padding=[0,1]) / (valid_mask_1 + 1e-4) * ((valid_mask_1 > 0).float() - valid_mask_0)
        valid_mask_num = (valid_mask_1 > 0).float().sum() - valid_mask_0.sum()
        valid_mask_0 = (valid_mask_1 > 0).float()

    return disp_init * valid_mask + disp * (1 - valid_mask)




class NLSPN(nn.Module):
    def __init__(self, args, ch_g, ch_f, k_g, k_f):
        super(NLSPN, self).__init__()

        # Guidance : [B x ch_g x H x W]
        # Feature : [B x ch_f x H x W]

        # Currently only support ch_f == 1
        assert ch_f == 1, 'only tested with ch_f == 1 but {}'.format(ch_f)

        assert (k_g % 2) == 1, \
            'only odd kernel is supported but k_g = {}'.format(k_g)
        pad_g = int((k_g - 1) / 2)
        assert (k_f % 2) == 1, \
            'only odd kernel is supported but k_f = {}'.format(k_f)
        pad_f = int((k_f - 1) / 2)

        self.args = args
        self.prop_time = self.args["model"]["nlspn"]["prop_time"]
        self.affinity = self.args["model"]["nlspn"]["affinity"]

        self.ch_g = ch_g
        self.ch_f = ch_f
        self.k_g = k_g
        self.k_f = k_f
        # Assume zero offset for center pixels
        self.num = self.k_f * self.k_f - 1
        self.idx_ref = self.num // 2

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            self.conv_offset_aff = nn.Conv2d(
                self.ch_g, 3 * self.num, kernel_size=self.k_g, stride=1,
                padding=pad_g, bias=True
            )
            self.conv_offset_aff.weight.data.zero_()
            self.conv_offset_aff.bias.data.zero_()

            if self.affinity == 'TC':
                self.aff_scale_const = nn.Parameter(self.num * torch.ones(1))
                self.aff_scale_const.requires_grad = False
            elif self.affinity == 'TGASS':
                self.aff_scale_const = nn.Parameter(self.args["model"]["nlspn"]["affinity_gamma"] * self.num * torch.ones(1))
                self.aff_scale_const.requires_grad = False
            else:
                self.aff_scale_const = nn.Parameter(torch.ones(1))
                self.aff_scale_const.requires_grad = False
        else:
            raise NotImplementedError

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64

    def _get_offset_affinity(self, guidance, confidence=None, rgb=None):
        B, _, H, W = guidance.shape

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            offset_aff = self.conv_offset_aff(guidance)
            o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)

            # Add zero reference offset
            offset = torch.cat((o1, o2), dim=1).view(B, self.num, 2, H, W)
            list_offset = list(torch.chunk(offset, self.num, dim=1))
            list_offset.insert(self.idx_ref,
                               torch.zeros((B, 1, 2, H, W)).type_as(offset))
            offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

            if self.affinity in ['AS', 'ASS']:
                pass
            elif self.affinity in ['TC', 'TGASS']:
                aff = torch.tanh(aff) / self.aff_scale_const
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Apply confidence
        # TODO : Need more efficient way
        if self.args["model"]["nlspn"]["conf_prop"]:
            list_conf = []
            offset_each = torch.chunk(offset, self.num + 1, dim=1)

            modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()

            for idx_off in range(0, self.num + 1):
                ww = idx_off % self.k_f
                hh = idx_off // self.k_f

                if ww == (self.k_f - 1) / 2 and hh == (self.k_f - 1) / 2:
                    continue

                offset_tmp = offset_each[idx_off].detach()

                # NOTE : Use --legacy option ONLY for the pre-trained models
                # for ECCV20 results.
                if self.args["model"]["nlspn"]["legacy"]:
                    offset_tmp[:, 0, :, :] = \
                        offset_tmp[:, 0, :, :] + hh - (self.k_f - 1) / 2
                    offset_tmp[:, 1, :, :] = \
                        offset_tmp[:, 1, :, :] + ww - (self.k_f - 1) / 2

                conf_tmp = ModulatedDeformConvFunction.apply(
                    confidence, offset_tmp, modulation_dummy, self.w_conf,
                    self.b, self.stride, 0, self.dilation, self.groups,
                    self.deformable_groups, self.im2col_step)
                list_conf.append(conf_tmp)

            conf_aff = torch.cat(list_conf, dim=1)
            aff = aff * conf_aff.contiguous()

        # Affinity normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

        if self.affinity in ['ASS', 'TGASS']:
            aff_abs_sum[aff_abs_sum < 1.0] = 1.0

        if self.affinity in ['AS', 'ASS', 'TGASS']:
            aff = aff / aff_abs_sum                        # 这里存在这么一种情况, 邻域权重全为负数, 绝对值求和为１, 则中心像素的权重恒为2, 将发生深度爆炸

        aff_sum = torch.sum(aff, dim=1, keepdim=True)      # 若保留gamma参数的可学习性, 且不对gamma进行人为约束, 则gamma会变0产生巨大的aff; 此外, 这一句也不符合spn绝对值求和小于1的要求
        aff_ref = 1.0 - aff_sum

        list_aff = list(torch.chunk(aff, self.num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)

        return offset, aff

    def _propagate_once(self, feat, offset, aff):
        feat = ModulatedDeformConvFunction.apply(
            feat, offset, aff, self.w, self.b, self.stride, self.padding,
            self.dilation, self.groups, self.deformable_groups, self.im2col_step
        )

        return feat

    def forward(self, feat_init, guidance, confidence=None, feat_fix=None,
                rgb=None):
        assert self.ch_g == guidance.shape[1]
        assert self.ch_f == feat_init.shape[1]

        if self.args["model"]["nlspn"]["conf_prop"]:
            assert confidence is not None

        if self.args["model"]["nlspn"]["conf_prop"]:
            offset, aff = self._get_offset_affinity(guidance, confidence, rgb)
        else:
            offset, aff = self._get_offset_affinity(guidance, None, rgb)

        # Propagation
        if self.args["model"]["nlspn"]["preserve_input"]:
            assert feat_init.shape == feat_fix.shape
            mask_fix = torch.sum(feat_fix > 0.0, dim=1, keepdim=True).detach()
            mask_fix = (mask_fix > 0.0).type_as(feat_fix)

        feat_result = feat_init

        list_feat = []

        for k in range(1, self.prop_time + 1):
            # Input preservation for each iteration
            if self.args["model"]["nlspn"]["preserve_input"]:
                feat_result = (1.0 - mask_fix) * feat_result \
                              + mask_fix * feat_fix

            feat_result = self._propagate_once(feat_result, offset, aff)

            list_feat.append(feat_result)

        # if feat_result.max() > 1e3:
        #     print("Error accure!")

        return feat_result, list_feat, offset, aff, self.aff_scale_const.data