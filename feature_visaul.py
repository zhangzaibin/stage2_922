from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from cv_networks import MCNet
from weight_manynetworks import ResnetEncoderMatching
from cvmap_networks import cvmap
from torch.autograd import Variable
from collections import OrderedDict
from IPython import embed
from options import MonodepthOptions
import matplotlib.pyplot as plt

my_device = torch.device("cuda:1")

# mcnet = MCNet(50,1.5).to(my_device)
# feature_ex = mcnet.feature_extraction
# # input = torch.Tensor(1, 3, 192, 640)
# # output = feature_ex(input)
# checkpoint = torch.load("./dpsnet_3_checkpoint.pth.tar")['state_dict']
#
# model_dict = {}
# state_dict = feature_ex.state_dict()
# for k, v in checkpoint.items():
#     if k.split('feature_extraction.')[-1] in state_dict:
#         model_dict[k.split('feature_extraction.')[-1]] = v
# state_dict.update(model_dict)
# feature_ex.load_state_dict(state_dict)
# feature_ex.to(my_device)
#
#
# for p in feature_ex.parameters():
#     p.requires_grad = False

fea_encoder = ResnetEncoderMatching(18, True, input_height=192, input_width=640, adaptive_bins=True, min_depth_bin=0.1,
                                    max_depth_bin=20, depth_binning='linear', num_depth_bins=96)
checkpoint = torch.load("./weight_pretrained/encoder.pth")
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    if k in fea_encoder.state_dict().keys():
        new_state_dict[k] = v

weight_decoder = fea_encoder.weight_decoder
weight_decoder.to(my_device)
print(fea_encoder)
feature_ex = nn.Sequential(fea_encoder.layer0, fea_encoder.layer1)
feature_ex.to(my_device)

for p in feature_ex.parameters():
    p.requires_grad = False

options = MonodepthOptions()
opts = options.parse()

datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                 "kitti_odom": datasets.KITTIOdomDataset}
dataset = datasets_dict[opts.dataset]

fpath = os.path.join(os.path.dirname(__file__), "splits", opts.split, "{}_files.txt")

train_filenames = readlines(fpath.format("train"))
val_filenames = readlines(fpath.format("val"))
img_ext = '.png' if opts.png else '.jpg'

num_train_samples = len(train_filenames)
num_total_steps = num_train_samples // opts.batch_size * opts.num_epochs

train_dataset = dataset(
    opts.data_path, train_filenames, opts.height, opts.width,
    opts.frame_ids, 4, is_train=True, img_ext=img_ext)
train_loader = DataLoader(
    train_dataset, opts.batch_size, True,
    num_workers=1, pin_memory=True, drop_last=True)
val_dataset = dataset(
    opts.data_path, val_filenames, opts.height, opts.width,
    opts.frame_ids, 4, is_train=False, img_ext=img_ext)
val_loader = DataLoader(
    val_dataset, opts.batch_size, shuffle=False,
    num_workers=1, pin_memory=True, drop_last=True)
val_iter = iter(val_loader)

for batch_idx, inputs in enumerate(val_loader):
    a1 = feature_ex(inputs[("color", 1, 0)].to(my_device))
    a2 = feature_ex(inputs[("color", 0, 0)].to(my_device))


    # fea = torch.cat((a2, a1), 1)
    # a = weight_decoder(fea)

    # print(a.shape)
    # print(a[3].max())
    # print(fea.shape)
    # b = a1[2].mean(0, True).squeeze(0)
    # b2 = a2[2].mean(0, True).squeeze(0)
    # b = a1[4][30].squeeze(0)
    # b2 = a2[0][30].squeeze(0)
    # b = a1[0][30].squeeze(0)




    # _b = a1*(a.reshape(12, 64, 1, 1))
    # print(_b[0].shape)
    # b = torch.sum(_b[2], 0)
    # print(b.shape)


    # b2 = a2[8][30].squeeze(0)
    # b = a1[8][30].squeeze(0)
    im_show = 8
    b2 = a2[im_show].mean(0, True)
    b = a1[im_show].mean(0, True)

    # im1 = (b.detach().cpu().numpy()-0.3)*100
    # im2 = (b2.detach().cpu().numpy()-0.3)*100
    b = b.permute(1, 2, 0)
    b2 = b2.permute(1, 2, 0)
    im1 = b.detach().cpu().numpy()
    im2 = b2.detach().cpu().numpy()

    print(im1.max())

    im = inputs[("color", 1, 0)][im_show].cpu()
    im = im.permute(1, 2, 0).numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(im1, vmin=0.4, vmax=0.8, cmap=plt.cm.jet)
    plt.subplot(1, 2, 2)
    plt.imshow(im2, vmin=0.4, vmax=0.8, cmap=plt.cm.jet)
    # plt.imshow(im)

    plt.show()
    break
