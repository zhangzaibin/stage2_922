from __future__ import print_function
import torch
import torch.nn as nn
from cv_networks.resnet_encoder import ResnetEncoder
from cv_networks.feature_decoder import DepthDecoder
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels= in_planes, out_channels= out_planes, kernel_size= kernel_size, stride= stride,
                                   padding= dilation if dilation >1 else pad, dilation= dilation),
                         nn.BatchNorm2d(out_planes))
def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride),
                         nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    """
    downsample: to distinguish the first block and the other block,it means xuxian he shixian
    """
    expansion = 1 #the mul between the two conv layers in the basicblock
    def __init__(self, in_planes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(in_planes, planes, 3, stride=stride, pad=pad, dilation=dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, stride=1, pad=pad, dilation=dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out






# class feature_extraction(nn.Module):
#     def __init__(self, num_layers=18, pretrained=True):
#         super(feature_extraction, self).__init__()
#         resnets = {18: models.resnet18,
#                    34: models.resnet34,
#                    50: models.resnet50,
#                    101: models.resnet101,
#                    152: models.resnet152}
#
#         if num_layers not in resnets:
#             raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
#
#         encoder = resnets[num_layers](pretrained)
#         self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
#         self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
#
#     def forward(self, x):
#         out = (x - 0.45) / 0.225
#         out = self.layer0(out)
#         out = self.layer1(out)
#         return out
class feature_extraction(nn.Module):
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(feature_extraction, self).__init__()
        self.encoder = ResnetEncoder(num_layers, pretrained)
        self.featureencoder = DepthDecoder(self.encoder.num_ch_enc)

    def forward(self, x):
        x = self.encoder(x)
        outputs = self.featureencoder(x)
        return outputs






class disparityregression(nn.Module):
    def __init__(self, depth_avaliable):
        super(disparityregression, self).__init__()
        # self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)
        self.disp = depth_avaliable
        # self.mindisp = mindisp
        # self.disp = (self.disp+1)*mindisp
        # self.disp[:, maxdisp-1, :, :] = 255


    def forward(self, x):
        # disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        # out = torch.sum(x*disp,1)
        _, idxs = torch.max(x, 1)
        out = self.disp[:, :, 2, :, :].squeeze(1) + (idxs - 2).float()

        return out

