from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import datasets
from cv_networks import MCNet
from kitti_utils import *
from layers import *
from options import MonodepthOptions
from utils import *

my_device = torch.device("cuda:1")

mcnet = MCNet(50,1.5).to(my_device)
feature_ex = mcnet.feature_extraction
checkpoint = torch.load("./dpsnet_8_checkpoint.pth.tar")['state_dict']

model_dict = {}
state_dict = feature_ex.state_dict()
for k, v in checkpoint.items():
    if k.split('feature_extraction.')[-1] in state_dict:
        model_dict[k.split('feature_extraction.')[-1]] = v
state_dict.update(model_dict)
feature_ex.load_state_dict(state_dict)
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





    im_show = 8
    # b2 = a2[4][30].squeeze(0)
    # b = a1[4][30].squeeze(0)
    b2 = a2[im_show].mean(0, True).squeeze(0)
    b = a1[im_show].mean(0, True).squeeze(0)
    # b2 = a2[im_show][40].squeeze(0)
    # b = a1[im_show][40].squeeze(0)

    im1 = b.detach().cpu().numpy()
    im2 = b2.detach().cpu().numpy()
    # im1 = (b.detach().cpu().numpy()-0.3)*100
    # im2 = (b2.detach().cpu().numpy()-0.3)*100

    # print(im1.max())

    im = inputs[("color", 1, 0)][im_show].cpu()
    im = im.permute(1, 2, 0).numpy()

    plt.subplot(1, 2, 1)
    # im2 = np.clip(im2, 0.2, 2)
    plt.imshow(im1, vmin=0.0, vmax=9, cmap=plt.cm.jet)
    # plt.subplot(1, 3, 2)
    # # plt.imshow(im2, vmin=-10, vmax=100, cmap=plt.cm.jet)
    # # plt.imshow(im1, vmin=-1, vmax=1, cmap=plt.cm.jet)
    # plt.imshow(im)
    plt.subplot(1, 2, 2)
    plt.imshow(im2, vmin=0.0, vmax=9, cmap=plt.cm.jet)
    # plt.imshow(im)
    plt.show()
    break
