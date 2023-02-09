'''
本文档参考Monodepth改写，并加入了所需要的有监督损失函数. 注意grid_sample的一些原理，避免出错
由于重构损失需要从深度中获取视差，进而获取重构视图，因此视差不可或缺
两个地方需要改动，包括将重构损失的超出视差位置mask掉
以及加入二阶平滑损失，注意说明原理，深度平面与相机平行、垂直（包括水平垂直和竖直方向的垂直两种情况）

另外，为了增强训练效果，需要检查PAM模块是否有需要改进的地方。

完成相关的可视化代码，包括中间结果的可视化。
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .torch_utils import tensor2img
import math

import cv2
import numpy as np


class DesignLoss(nn.modules.Module):
    def __init__(self, args):
        super(DesignLoss, self).__init__()
        self.predict_w = args["loss"]["predict_weight"]            # 预测深度偏差权重
        self.SSIM_w    = args["loss"]["SSIM_weight"]               # SSIM项的权重
        self.smooth_w  = args["loss"]["smooth_weight"]             # 视差/深度平滑权重
        self.photo_w   = args["loss"]["photometric_weight"]        # 光度重构权重
        self.PAM_w     = args["loss"]["PAM_weight"]                # PAM自监督权重

        self.max_disp  = args["model"]["max_disp"]
        self.max_depth = args["model"]["max_depth"]

        self.loss_dict = {}

    # warp function block
    def scale_transform(self, img, ratio):
        # 视差图下采样
        _, _, h, w = img.size()                     # BCHW
        h = h // ratio
        w = w // ratio
        scaled_img = nn.functional.interpolate(img,
                                               size=[h, w], mode='bilinear',
                                               align_corners=True)
        return scaled_img

    def apply_disparity(self, img, disp):
        # 应用视差和图像生成另一个视角的视图
        # disp in (1, width)
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        # 采用(0,1)的归一化坐标是因为有多个尺度，将归一化坐标乘以缩放因子和实际图像尺寸即可恢复原图像坐标
        # x_base 和 y_base为img同size的坐标
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        # x_base, y_base, x_shift 需要全部归一化到(0,1)内
        x_shifts = disp[:, 0, :, :] / width # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)    # B * H * W * 2 ，(x坐标+偏置, y坐标)
        # In grid_sample coordinates are assumed to be between -1 and 1 :
        # 2*flow_field - 1为坐标变换，将(0,1)的坐标线性映射到(-1,1)
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros', align_corners=False)
        valid = self.get_mask(flow_field, occlusion=False)

        return output.contiguous(), valid.contiguous()

    def get_mask(self, flow_field, occlusion=False):
        '''
        flow_field: B * H * W * 2, (x, y)
        mask: valid area=1, else=0
        use for generate left from right.
        '''
        B, H, W, _ = flow_field.shape
        flow_field = flow_field.permute(0, 3, 1, 2).contiguous().view(B, 2, H, W)    # B 2 H W
        mask = torch.ones([B, 1, H, W]).to(flow_field.device) * (flow_field[:,:-1,:,:] >= 0).float() * (flow_field[:,:-1,:,:] <= 1).float()
        if occlusion:
            for i in range(1, W):
                pad_field = flow_field[:, :-1, :, i:]
                pad_field = F.pad(pad_field, (0, i, 0, 0), 'constant', 0)       # pad(left, right, up, down)
                temp = (torch.abs(flow_field[:,:-1,:,:] - pad_field) > 1/(2*W)).float()
                mask = mask * temp
        return mask.detach()


    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    # smooth loss block
    def gradient_x(self, img):
        # 计算水平方向梯度，pad用以保持图像尺寸
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")    # 右填充，相同元素
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]           # NCHW
        return gx

    def gradient_y(self, img):
        # 计算垂直方向梯度，pad用以保持图像尺寸
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")    # 下填充，相同元素
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]           # NCHW
        return gy

    def disp_smoothness(self, disp, img, vis=False):
        # disp左视差图，img左RGB，BCHW

        # 根据坤洪师兄的建议，采用一阶和二阶平滑损失共同训练

        # 视差图梯度 B1HW
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        # 视差图的二阶梯度 B1HW
        disp_gradients_gradients_x = self.gradient_x(disp_gradients_x)
        disp_gradients_gradients_y = self.gradient_y(disp_gradients_y)

        # RGB左图梯度 B3HW
        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        # 梯度计算方法为RGB三个通道的梯度平均
        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

        smoothness_L1 = torch.abs(disp_gradients_x * weights_x) + torch.abs(disp_gradients_y * weights_y)
        smoothness_L2 = torch.abs(disp_gradients_gradients_x * weights_x) + \
                        torch.abs(disp_gradients_gradients_y * weights_y)

        return smoothness_L1 + smoothness_L2

    # ssim loss block
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map

    def ssim(self, img1, img2, window_size=11):
        _, channel, h, w = img1.size()
        window = self.create_window(window_size, channel)
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        return self._ssim(img1, img2, window, window_size, channel)


    def forward(self, disp_list, Valid_left_to_right, gt_depth, cal, PAM_list, imgs, vis=False):
        # 前向传播计算loss
        # pt_disp:预测视差 B 1 H W
        # gt_depth:深度真值 B 1 H W
        # cal:包含参数baseline&focal B 2
        # PAM_list:PAM用以自监督的项
        # img:(imgL, imgR)
        # vis: visualize
        # 网络预测为深度，监督为深度

        PAM_disp, Mask_disp, SPN_inter, pt_disp = disp_list

        # 根据预测深度与基线、焦距，计算视差。注意先将预测的深度约束在 (1, 100)，使深度和视差的关系约束在近似线性关系内
        # 避免零映射带来的数值波动
        # disp -> (0, 192); depth -> (0, 100); 取中间映射区间 注意映射区间和图像的尺寸有关，这里的尺寸是原始输入尺寸
        # depth = (1, 100) <==> disp = (3.9 ,390)
        imgL, imgR = imgs
        focal = cal[:,1][:,None,None,None].type_as(imgs[0])
        baseline = cal[:,0][:,None,None,None].type_as(imgs[0])
        gt_dense = (focal*baseline) / (gt_depth + 1e-8)                     # 获取视差的标签
        pt_disp_temp = pt_disp.data.clamp(1, self.max_disp) + (pt_disp - pt_disp.data)
        pt_depth = (focal*baseline) / pt_disp_temp         # 比较深度的预测效果

        ####### 直接深度预测损失
        # 考虑到误差较大的点比较多，因此将平滑L1换成RMSE，
        # 一方面是增大对较大错误预测点的乘法力度，另一方面是与KITTI评测一致
        # 这里不将预测结果进行范围约束，保证网络预测的深度可以尽可能接近真实深度
        mask0 = (gt_depth > 1e-4).detach_().float()             # gt区域
        mask1 = (Valid_left_to_right > 0.5).float() * mask0     # 非遮挡且有gt区域
        mask2 = (Valid_left_to_right < 0.5).float() * mask0     # 遮挡且有gt区域

        predict_loss = 0.5 * F.smooth_l1_loss(PAM_disp[mask1.bool()], gt_dense[mask1.bool()]) + \
                       0.7 * (0.6 * F.smooth_l1_loss(Mask_disp[mask1.bool()], gt_dense[mask1.bool()]) + 
                              0.4 * F.smooth_l1_loss(Mask_disp[mask2.bool()], gt_dense[mask2.bool()])) + \
                       1.0 * F.mse_loss(pt_depth[mask0.bool()], gt_depth[mask0.bool()])

        ####### PAM自监督损失
        (M_right_to_left, M_left_to_right), \
        (M_left_right_left, M_right_left_right), \
        (V_left_to_right, V_right_to_left) = PAM_list

        # PAM算法中的平滑损失(相邻像素在变换矩阵上尽可能相同)
        loss_h = F.l1_loss(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                 F.l1_loss(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
        loss_w = F.l1_loss(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                 F.l1_loss(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
        PAM_smooth = loss_h + loss_w

        # PAM算法中的循环一致性损失(L -> R' -> L' == L; R -> L' -> R' == R)
        # 仅考虑第一次变换的mask，因为变换后与mask相乘，已经去除了遮挡部位
        # M_1_2_1 : B * H * W1 * W1; V_1_to_2 : B * 1 * H * W1
        B, C, H, W = imgL.shape
        H = int(H / 4)          # PAM中feature map的size是原图的1/4
        W = int(W / 4)
        Identity   = Variable(torch.eye(W, W).repeat(B, H, 1, 1), requires_grad=False).cuda()       # 单位矩阵
        PAM_cycle = F.l1_loss(M_left_right_left * V_left_to_right.permute(0, 2, 1, 3),  Identity * V_left_to_right.permute(0, 2, 1, 3)) + \
                    F.l1_loss(M_right_left_right * V_right_to_left.permute(0, 2, 1, 3), Identity * V_right_to_left.permute(0, 2, 1, 3))

        # PAM算法中的光度一致性损失
        mini_imgL = self.scale_transform(imgL, 4)
        mini_imgR = self.scale_transform(imgR, 4)

        LR_right_warped = torch.bmm(M_right_to_left.contiguous().view(B*H, W, W), mini_imgR.permute(0, 2, 3, 1).contiguous().view(B*H, W, C))
        LR_right_warped = LR_right_warped.view(B, H, W, C).contiguous().permute(0, 3, 1, 2)
        LR_left_warped  = torch.bmm(M_left_to_right.contiguous().view(B*H, W, W), mini_imgL.permute(0, 2, 3, 1).contiguous().view(B*H, W, C))
        LR_left_warped  = LR_left_warped.view(B, H, W, C).contiguous().permute(0, 3, 1, 2)

        PAM_photo = F.l1_loss(mini_imgL * V_left_to_right, LR_right_warped * V_left_to_right) + \
                    F.l1_loss(mini_imgR * V_right_to_left, LR_left_warped * V_right_to_left)

        PAM_loss = PAM_smooth + PAM_cycle + PAM_photo

        # 获取PAM的视差结果，该计算仅用于可视化的生成
        if vis:
            # 生成左图坐标
            right_position = Variable(torch.FloatTensor([i for i in range(0, W)]).repeat(B, 1, H, 1), requires_grad=False).cuda()
            tensor2img((right_position / W).detach()[0], 'PAM-right-position.png', colourize=True)    # 注意进行归一化处理

            # 生成右图坐标
            left_position = torch.bmm(M_right_to_left.contiguous().view(B*H, W, W), right_position.permute(0, 2, 3, 1).contiguous().view(B*H, W, 1))
            left_position = left_position.view(B, H, W, 1).contiguous().permute(0, 3, 1, 2)   # 获得的视差图的取值范围在 0-W
            tensor2img((left_position / W).detach()[0], 'PAM-left-position.png', colourize=True)

            # 计算视差 = 右图坐标 - 左图坐标
            PAM_disparity = right_position - left_position
            tensor2img((PAM_disparity / (192/4)).detach()[0], 'PAM-disparity.png', colourize=True)

            # # 观察降分辨率图像、PAM的生成图像是否存在问题
            tensor2img(mini_imgL.detach()[0], 'PAM-mini-left.png', ReNormalize=True)
            tensor2img(mini_imgR.detach()[0], 'PAM-mini-right.png', ReNormalize=True)
            tensor2img(LR_right_warped.detach()[0], 'PAM-mini-left-softG.png', ReNormalize=True)
            tensor2img(LR_left_warped.detach()[0], 'PAM-mini-right-softG.png', ReNormalize=True)
            tensor2img(torch.abs(mini_imgL - LR_right_warped).detach()[0], 'PAM-mini-left-loss.png', colourize=True)
            tensor2img(torch.abs(mini_imgR - LR_left_warped).detach()[0], 'PAM-mini-right-loss.png', colourize=True)

            # 可视化某个batch的某个Height的对应矩阵图
            tensor2img(M_right_to_left.detach()[0][H//4*1:H//4*1+1, :, :], 'PAM-M-R2L-01top.png', colourize=True)
            tensor2img(M_right_to_left.detach()[0][H//4*2:H//4*2+1, :, :], 'PAM-M-R2L-02center.png', colourize=True)
            tensor2img(M_right_to_left.detach()[0][H//4*3:H//4*3+1, :, :], 'PAM-M-R2L-03bottom.png', colourize=True)
            tensor2img(V_left_to_right.detach()[0], 'PAM-V-R2L.png')



        ###### photometric 重构损失 ######
        left_generate, valid = self.generate_image_left(imgR, pt_disp)       # 网络仅预测左图深度/视差，则仅考虑生成右图
        # 计算L1-loss, 采用 valid 排除遮挡区域
        l1_photometric = torch.abs(imgL * valid * Valid_left_to_right - left_generate * valid * Valid_left_to_right)
        # 计算 SSIM, 注意 SSIM 的计算结果图比原图少边界一圈，对 valid 进行 crop 处理, 使张量尺寸一致
        ssim_right = self.ssim(imgL, left_generate)
        ssim_right = ((1 - ssim_right) / 2) * valid * Valid_left_to_right



        ## 可视化效果，只看batch=0的结果
        if vis:
            # 这里生成的1/4分辨率图像的warp图只是用来可视化，到时候可以删除
            left_hard_G, left_hard_G_mask = self.generate_image_left(mini_imgR, PAM_disparity)
            tensor2img(imgL.detach()[0], 'Image-left.png', ReNormalize=True)
            tensor2img(imgR.detach()[0], 'Image-right.png', ReNormalize=True)
            tensor2img(left_generate.detach()[0], 'Image-left-warp.png', ReNormalize=True)

            tensor2img(l1_photometric.detach()[0], 'Image-L1-loss.png', colourize=True)
            tensor2img(ssim_right.detach()[0], 'Image-ssim.png', colourize=True)
            tensor2img((valid * Valid_left_to_right).detach()[0], 'Image-ssim-mask.png')

            tensor2img(pt_depth.detach()[0] /100, 'Image-left-depth.png', colourize=True)
            tensor2img(left_hard_G.detach()[0], 'PAM-mini-left-hardG.png', ReNormalize=True)
            tensor2img(torch.abs(mini_imgL - left_hard_G).detach()[0], 'PAM-mini-left-hard-loss.png', colourize=True)

        ## 计算loss
        l1_photometric = torch.mean(l1_photometric[(valid * Valid_left_to_right).repeat(1, 3, 1, 1).bool()])            # L1 photometric
        ssim_right = torch.mean(ssim_right[(valid * Valid_left_to_right).repeat(1, 3, 1, 1).bool()])                    # SSIM photometric
        photometric_loss = self.SSIM_w * ssim_right + (1 - self.SSIM_w) * l1_photometric



        ###### Disparities smoothness 平滑损失 ######
        disp_smoothness  = self.disp_smoothness(pt_disp, imgL)
        # depth_smoothness = self.disp_smoothness(pt_depth.clamp(1,100), imgL)
        smooth_loss = torch.mean(torch.abs(disp_smoothness)) #/ 2 + \
                      # torch.mean(torch.abs(depth_smoothness)) / 2



        ###### 所有 Loss 加权 ######
        loss =   self.predict_w * predict_loss\
               + self.smooth_w  * smooth_loss\
               + self.photo_w   * photometric_loss\
               + self.PAM_w     * PAM_loss

        # if loss>10:
        #     print(loss)

        self.loss_dict['predict_loss'] = predict_loss
        self.loss_dict['smooth_loss']  = smooth_loss
        self.loss_dict['photometric_loss'] = photometric_loss
        self.loss_dict['PAM_loss'] = (PAM_smooth, PAM_cycle, PAM_photo)
        
        return loss
