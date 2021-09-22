import torch
import torch.nn as nn
import math
from cv_networks.submodule import *
from .layers import *
import gc
import numpy as np


class MCNet(nn.Module):
    """
    stage 1 network use the contrast loss
    """
    def __init__(self, nlabel, num_gaussian, sampling_way):
        """

        :param nlabel: num of sampling points around the epipolar line
        :param num_gaussian: num of gaussian sample points
        :param sampling_way: the sample way:linear/nonlinear
        """
        super(MCNet, self).__init__()
        self.nlabel = nlabel
        self.gaussion_label = num_gaussian
        self.sampling_way = sampling_way
        self.feature_extraction = feature_extraction(num_layers=18, pretrained=True)


    # caculate the smooth loss dis and cvt
    def get_smooth_loss(self, disp, img):
        b, _, h, w = disp.size()
        img = F.interpolate(img, (h, w), mode='area')

        disp_dx, disp_dy = self.gradient(disp)
        img_dx, img_dy = self.gradient(img)

        disp_dxx, disp_dxy = self.gradient(disp_dx)
        disp_dyx, disp_dyy = self.gradient(disp_dy)

        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)

        smooth1 = torch.mean(disp_dx.abs() * torch.exp(-img_dx.abs().mean(1, True))) + \
                  torch.mean(disp_dy.abs() * torch.exp(-img_dy.abs().mean(1, True)))

        smooth2 = torch.mean(disp_dxx.abs() * torch.exp(-img_dxx.abs().mean(1, True))) + \
                  torch.mean(disp_dxy.abs() * torch.exp(-img_dxy.abs().mean(1, True))) + \
                  torch.mean(disp_dyx.abs() * torch.exp(-img_dyx.abs().mean(1, True))) + \
                  torch.mean(disp_dyy.abs() * torch.exp(-img_dyy.abs().mean(1, True)))

        return -1.0 * smooth1 + smooth2

    def gradient(self, D):
        dy = D[:, :, 1:] - D[:, :, :-1]
        dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return dx, dy

    def compute_contrast_loss(self, target_volumes, ref2_fea):
        """
        we use the triplet loss to increase the distance of the negative sample
        and reduce the distance of the positive sample
        """

        triplet_loss = torch.nn.TripletMarginLoss(margin=1, p=2, reduction = 'none')
        # every positive sample are relate to a anchor and positive sample
        tr_loss = 0.
        for i in range(self.nlabel+self.gaussion_label):
            if i == int(self.nlabel/2):
                continue
            tr_loss += triplet_loss(ref2_fea, target_volumes[:, :, int(self.nlabel/2), :, :], target_volumes[:, :, i, :, :])
        tr_loss /= (self.nlabel + self.gaussion_label - 1)

        return tr_loss




    def forward(self, ref2, targets, intrinsics, intrinsics_inv, ref_depth, real_img):
        """
        :param ref2:reference image usually is the later image(T2) to satisfy T1*T2-1
        :param targets1: original image is the former one(T1)
        :param intrinsics:
        :param intrinsics_inv:
        :return:
        """
        total_out = {}
        total_smooth_loss = {}
        ref_depth = F.interpolate(ref_depth, [48, 160], mode="bilinear", align_corners=True)
        # we use the multi-scale method to couple with the stage2 monodepth2 method

        self.feature_extraction.to(torch.device(ref2.device))
        ref2_fea_out = self.feature_extraction(ref2)[("f_disp", 0)]
        real_img_fea = self.feature_extraction(real_img)[("f_disp", 0)]
        fea_out = (ref2_fea_out, real_img_fea)
        for n in range(4):

            ref2_fea = self.feature_extraction(ref2)[("f_disp", n)]# the feature of ref image

            ref2_fea = F.interpolate(ref2_fea, [48, 160], mode="bilinear", align_corners=True)

            for j, target in enumerate(targets):


                # gaussion sample self.gaussion_label points around the gt depth
                cost = Variable(torch.FloatTensor(ref2_fea.size()[0], 1, self.nlabel+self.gaussion_label, ref2_fea.size()[2],ref2_fea.size()[3]).zero_()).cuda()
                target_volume = Variable(torch.FloatTensor(ref2_fea.size()[0], 16, self.nlabel+self.gaussion_label, ref2_fea.size()[2],ref2_fea.size()[3]).zero_()).cuda()
                # depth_avaliable = Variable(torch.FloatTensor(ref2_fea.size()[0], 1, self.nlabel, 192,640).zero_()).cuda()
                target_fea = self.feature_extraction(target[0].cuda())[("f_disp", n)]#we set the specific device:1 here it's not cool!
                target_fea = F.interpolate(target_fea, [48, 160], mode="bilinear", align_corners=True)
                for i in range(self.nlabel):
                    # i begin from 0 to 9, so we set i+1
                    _depth = torch.clamp((ref_depth + (i - int(self.nlabel/2))).float().cpu(), 0, 255.0)#we don't set the max depth 75 it's too short
                    # depth = F.interpolate(_depth, [48, 160], mode='bilinear').squeeze(1)
                    depth = _depth.squeeze(1)
                    # depth_avaliable[:, 0, i, :, :] = _depth.squeeze(1)
                    # depth = (Variable(torch.ones(ref2_fea.size(0), ref2_fea.size(2), ref2_fea.size(3)))* self.mindepth * (i+1)).float().cpu()#convert the disp to the depth
                    back_pro_depth = BackprojectDepth(target_fea.size(0), height=target_fea.size()[2], width=target_fea.size()[3]).cpu()#image to point cloud
                    depth_cloud = back_pro_depth(depth, intrinsics_inv).cpu()
                    project_3d = Project3D(batchsize=target_fea.size(0), width=target_fea.size()[3], height=target_fea.size()[2]).cpu()
                    p_coord = project_3d(depth_cloud, intrinsics.cpu(), target[1].cpu()).cuda()

                    # get the gt depth p_coord
                    if i == int(self.nlabel/2):
                        _p_coord = p_coord

                    #解决程序训练到第三个epoch被杀死的问题
                    del depth_cloud, project_3d
                    gc.collect()
                    targetimg_fea_w = F.grid_sample(target_fea, p_coord, mode= 'bilinear', padding_mode='border')# the feature map in the epipolar line
                    targetimg_fea_w = targetimg_fea_w
                    # cost[:, 0, i, :, :] = torch.cat((ref2_fea, targetimg_fea_w), dim=1)
                    cost[:, 0, i, :, :] = torch.mean(torch.abs(ref2_fea - targetimg_fea_w), dim=1)
                    target_volume[:, :, i, :, :] = targetimg_fea_w

                    # triger the gaussion sample operation
                    if i == self.nlabel - 1:
                        cost[:, 0, i, :, :] = torch.mean(torch.abs(ref2_fea - targetimg_fea_w), dim=1)
                        for w in range(self.gaussion_label):
                            p_coord_g = _p_coord + torch.tensor(
                                np.random.normal(0, 0.01, p_coord.size())).float().cuda()
                            targetimg_fea_w = F.grid_sample(target_fea, p_coord_g, mode='bilinear', padding_mode='border')
                            targetimg_fea_w =  targetimg_fea_w

                            cost[:, 0, i + w + 1, :, :] = torch.mean(torch.abs(ref2_fea - targetimg_fea_w), dim=1)
                            target_volume[:, :, i + w + 1, :, :] = targetimg_fea_w

                if j == 0:
                    target_volumes = target_volume
                    costs = cost
                else:
                    target_volumes = target_volumes + target_volume
                    costs = costs + cost

            target_volumes = target_volumes/len(targets)
            costs = costs/len(targets)
            out = self.compute_contrast_loss(target_volumes, ref2_fea)
            total_out[('f_disp', n)] = out
            # smooth_loss = self.get_smooth_loss(disp=ref2_fea, img=ref2)
            # total_smooth_loss[('f_disp', n)] = smooth_loss

        return total_out, fea_out, costs[:, 0, int(self.nlabel/2), :, :], ref2_fea





