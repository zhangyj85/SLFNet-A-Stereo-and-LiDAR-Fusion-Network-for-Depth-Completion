import torch
import cv2
import os
import numpy as np

def model_info(model, verbose=False):
    # Model information.
    n_p = sum(x.numel() for x in model.parameters())                        # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)     # number gradients
    
    if verbose:
        print('%5s %60s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            print('%5g %60s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

def tensor2img(tensor, filename, ReNormalize=False, colourize=False, mask=False):
    # convert tensor to image and save
    # tensor = C H W, 数值范围(0, 1)
    savedir = './vis_imgs'
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    output = tensor.permute(1,2,0).cpu().numpy()

    if ReNormalize:
        assert (output.shape[2] == 3), "ReNormalize failed! Please check the tensor channels."
        # output = H W C, mean = (3,) std = (3,)
        output = output * np.array(std)[None, None, :] + np.array(mean)[None, None, :]
        # (0, 1) -> (0, 255)
        output = (255 * output).astype(np.uint8)
        # RGB -> BGR
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    elif colourize:
        # 保留mask, 便于后期处理
        valid = output >= 0
        output = (255 * output).astype(np.uint8)

        if output.shape[2] == 1:
            print("Convert Gray Image to pseudo-Color Image.")
        else:
            print("Convert Color Image to pseudo-Color Image.")
            output = np.mean(output, axis=2, keepdims=True)

        # 固定alpha, 从而便于定色比较距离
        alpha = 1.0
        # 将灰度图转为 BGR 伪彩色
        output = cv2.applyColorMap(cv2.convertScaleAbs(output, alpha=alpha), cv2.COLORMAP_JET)
        if mask:
            output = output * valid.astype(np.uint8)

    else:
        output = (255 * output).astype(np.uint8)

    cv2.imwrite(os.path.join(savedir, filename), output)

    return output



'''
将深度图转为点云
'''
class Dep2PcdTool(object):
    """docstring for Dep2PcdTool"""
    def __init__(self):
        super(Dep2PcdTool, self).__init__()

    def renormalize(self, color):
        """
        input:  bchw, torch.normalize 之后的结果
        output: bhwc, (0,1)
        """
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        color = color * std[None, :, None, None].cuda() + mean[None, :, None, None].cuda()
        return color.permute(0,2,3,1) * 255

    def pcd2ply(self, rgb, dep, calib, ply_file):
        """
        dep: numpy array (H, W, 1), 0-100
        rgb: numpy array (H, W, 3), 0-255
        """
        rgb = np.array(rgb, dtype="float32")
        dep = np.array(dep, dtype="float32")
        pcd = self.rgbd2pcd(rgb, dep, calib)
        # f"{}" replace the contain with variable
        header = "ply\n" + \
                 "format ascii 1.0\n" + \
                 f"element vertex {pcd.shape[0]}\n" +\
                 "property float32 x\n" + \
                 "property float32 y\n" + \
                 "property float32 z\n" + \
                 "property uint8 red\n" + \
                 "property uint8 green\n" + \
                 "property uint8 blue\n" + \
                 "end_header\n"
        with open(ply_file, 'w+') as f:
            f.write(header)
            for i in range(pcd.shape[0]):
                x, y, z, r, g, b = pcd[i,:]
                line = '{:.5f} {:.5f} {:.5f} {:.0f} {:.0f} {:.0f}\n'.format(x,y,z,r,g,b)
                f.write(line)

    def rgbd2pcd(self, rgb, dep, calib):
        """
        rgb: numpy array (H, W, 3), (0,1)
        dep: numpy array (H, W, 1), (0,192)
        pcd: numpy array (N, 6)
        """
        xyz = self.dep2xyz(dep, calib)  # (N, 3), N=HW
        rgb = rgb.reshape(-1, 3)        # (N, 3)
        pcd = np.concatenate([xyz, rgb], axis=1)
        return pcd                      # (N, 6)

    def dep2xyz(self, dep, calib):
        """
        dep: numpy.array (H, W, 1)
        cal: numpy.array (3, 3)
        xyz: numpy.array (N, 3)
        """
        # 生成图像坐标
        u, v = np.meshgrid(np.arange(dep.shape[1]), np.arange(dep.shape[0]))    # (H, W, 2)
        u, v = u.reshape(-1), v.reshape(-1)                                     # (H*W,), (H*W,)

        # 构成所需的坐标矩阵
        img_coord = np.stack([u, v, np.ones_like(u)])   # (3, H*W), (u,v,1)
        # print(calib)
        cam_coord = np.linalg.inv(calib) @ img_coord    # (3,3)^(-1) * (3, HW)
        xyz_coord = cam_coord * dep[v, u, 0]            # (3, HW)
        return xyz_coord.T                              # (HW, 3)

"""
用法:
cc = torch.tensor([7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03])
cc = cc.reshape(3,4)
cc = cc[:, :3]
cc = np.array(cc)

tool = Dep2PcdTool()
tool.pcd2ply(tool.renormalize(imgL)[0].data.cpu(), cal[0,0].data.cpu() * cal[0,1].data.cpu() / sparseL[0].clamp(1,192).permute(1,2,0).data.cpu(), cc, "/home/kunb/KITTI_DepthComplement/depth/model_draw/paper_figs/testtool2.ply")
"""