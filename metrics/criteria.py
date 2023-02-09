#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    criteria.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/14 7:51 PM

import torch
import torch.nn as nn

class Metrics(nn.Module):

    def __init__(self, args):
        super(Metrics, self).__init__()
        self.min_depth = 1e-4
        self.max_depth = args["model"]["max_depth"]

    def forward(self, pt, gt, *args):
        with torch.no_grad():
            pt = pt.detach()
            gt = gt.detach()

            pt_inv = 1.0 / (pt + 1e-8)
            gt_inv = 1.0 / (gt + 1e-8)

            # For numerical stability
            mask = (gt > self.min_depth).float() * (gt < self.max_depth).float()
            mask = mask.bool()
            num_valid = mask.sum()

            pt = pt[mask]
            gt = gt[mask]

            pt_inv = pt_inv[mask]
            gt_inv = gt_inv[mask]

            # pt_inv[pt <= self.thresh] = 0.0
            # gt_inv[gt <= self.thresh] = 0.0

            # RMSE / MAE
            diff = pt - gt
            diff_abs = torch.abs(diff)
            diff_sqr = torch.pow(diff, 2)

            rmse = diff_sqr.sum() / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = diff_abs.sum() / (num_valid + 1e-8)

            # iRMSE / iMAE
            diff_inv = pt_inv - gt_inv
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = torch.pow(diff_inv, 2)

            irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = diff_inv_abs.sum() / (num_valid + 1e-8)

            # Rel
            rel = diff_abs / (gt + 1e-8)
            rel = rel.sum() / (num_valid + 1e-8)

            # delta
            r1 = gt / (pt + 1e-8)
            r2 = pt / (gt + 1e-8)
            ratio = torch.max(r1, r2)

            del_1 = (ratio < 1.25).type_as(ratio)
            del_2 = (ratio < 1.25**2).type_as(ratio)
            del_3 = (ratio < 1.25**3).type_as(ratio)

            del_1 = del_1.sum() / (num_valid + 1e-8)
            del_2 = del_2.sum() / (num_valid + 1e-8)
            del_3 = del_3.sum() / (num_valid + 1e-8)

            result = [rmse, mae, irmse, imae, rel, del_1, del_2, del_3]
            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()

        return result