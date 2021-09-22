from __future__ import absolute_import, division, print_function

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F



class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)



class BackprojectDepth(nn.Module):
    """
    layer to transform the depth to the 3D point cloud
    """
    def __init__(self, batchsize, height, width):
        super(BackprojectDepth, self).__init__()
        self.batchsize = batchsize
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batchsize, 1, self.width*self.height),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), dim=0)
        self.pix_coords = self.pix_coords.repeat(self.batchsize, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)
        """
        the shape of pix_coords is like [x1, x2]
                          [y1, y2]
                          [1,   1]
        it's the same with the formula format"""

    def forward(self, depth, inv_K):
        """
        :param depth: width*height
        :param inv_K: batchsize*3*3
        :return:
        """

        cam_points = torch.matmul(inv_K.cpu(), self.pix_coords)
        cam_points = depth.view(self.batchsize, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    def __init__(self, batchsize, width, height, eps=1e-7):
        super(Project3D, self).__init__()
        self.batchsize = batchsize
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        """
        :param cam_points:batchsize * 4 * (width*height)
        :param K: shape:batchsize * 3 * 3
        :param T: shape:batchsize * 3 * 4
        :return: pix_coords
        """
        #P:b * 3 * 4
        P1 = torch.matmul(T, points)

        cam_points = torch.matmul(K, P1)

        #pix_coords is the coords of the
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batchsize, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width-1
        pix_coords[..., 1] /= self.height-1
        """
        the pix_coords's shape is like
        [[x1, y1], [x2, y2], [x3, y3]]
        [[x4, y4]....................]
        ............................."""
        pix_coords = (pix_coords - 0.5) * 2
        #transform the O point to the centor of image but why to *2
        return pix_coords




class ConvBlock(nn.Module):

    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        # out = self.bn(out)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class CABlock(nn.Module):
    """
    this is the channel_wise attention block
    first we use the senet idea to focus the channel wise information
    https://github.com/moskomule/senet.pytorch
    in_channels: number of input feature map channels
    out_channels: number of output feature channels default 16
    """
    def __init__(self, in_channels, out_channels):
        super(CABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.out_channels = out_channels
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16,  out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_channels, 1, 1)
        return y


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="bilinear")


