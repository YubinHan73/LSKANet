# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .fcn_head import FCNHead
import cv2
import time
import numpy as np
import os
#
#
# def save_image(tensor):
#     dir = '/data2/hyb/SegNetwork_other/SegNeXt-main/tools/work_dirs/0_LSKANet/LSKANet_hr48_cadist3_new/bm/'
#     featmap = tensor.cpu().clone()  # we clone the tensor to not do changes on it
#     feat_map = torch.mean(featmap, dim=0)
#     norm_img = np.zeros(feat_map.shape)
#     norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
#     norm_img = np.asarray(norm_img, dtype=np.uint8)
#     image = image.squeeze(0)  # remove the fake batch dimension
#     image = unloader(image)
#     if not os.exists(dir):
#         os.makedirs(dir)
#     image.save('results_{}.jpg'
#                .format(num, para['style_weight'], para['content_weight'], para['lr'], para['epoch'],
#                        para['style_loss'], para['content_loss']))

@HEADS.register_module()
class BGHead(FCNHead):

    def __init__(self, boundary_threshold=0.1, **kwargs):
        super(BGHead, self).__init__(**kwargs)
        self.boundary_threshold = boundary_threshold
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1],
                         dtype=torch.float32,
                         requires_grad=False).reshape((1, 1, 3, 3)))
        self.fusion_kernel = torch.nn.Parameter(
            torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                         dtype=torch.float32).reshape(1, 3, 1, 1),
            requires_grad=False)

    def losses(self, seg_logit, seg_label):
        """Compute Boundary Loss."""
        seg_label = seg_label.to(self.laplacian_kernel)

        boundary_targets = F.conv2d(
            seg_label, self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > self.boundary_threshold] = 1
        boundary_targets[boundary_targets <= self.boundary_threshold] = 0

        boundary_targets_x2 = F.conv2d(
            seg_label, self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)
        boundary_targets_x2_up = F.interpolate(
            boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up[
            boundary_targets_x2_up > self.boundary_threshold] = 1
        boundary_targets_x2_up[
            boundary_targets_x2_up <= self.boundary_threshold] = 0

        boundary_targets_x4 = F.conv2d(
            seg_label, self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)
        boundary_targets_x4_up = F.interpolate(
            boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up[
            boundary_targets_x4_up > self.boundary_threshold] = 1
        boundary_targets_x4_up[
            boundary_targets_x4_up <= self.boundary_threshold] = 0

        boundary_targets_pyramids = torch.stack(
            (boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
        boundary_targets_pyramids = boundary_targets_pyramids.squeeze(2)
        boundary_targets_pyramid = F.conv2d(boundary_targets_pyramids, self.fusion_kernel)
        boundary_targets_pyramid[
            boundary_targets_pyramid > self.boundary_threshold] = 1
        boundary_targets_pyramid[
            boundary_targets_pyramid <= self.boundary_threshold] = 0

        loss = super(BGHead, self).losses(seg_logit, boundary_targets_pyramid.long())

        return loss
