from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import datasets
from cv_networks import MCNet
from kitti_utils import *
from layers import *
from options import MonodepthOptions
from utils import *
import PIL.Image as Image
from torchvision import transforms

my_device = torch.device("cuda:0")

mcnet = MCNet(11 , 5, 'linear').to(my_device)
feature_ex = mcnet.feature_extraction
# print(feature_ex)
checkpoint = torch.load("./stage1_pretrained/stage1_gan/dpsnet_3_checkpoint.pth.tar")['state_dict']
#checkpoint = torch.load("./dpsnet_3_checkpoint.pth.tar")['state_dict']

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


# CA_Net = mcnet.CA_Block
#
# c_model_dict = {}
# c_state_dict = CA_Net.state_dict()
# for k, v in checkpoint.items():
#     if k.split('CA_Block.')[-1] in c_state_dict:
#         c_model_dict[k.split('CA_Block.')[-1]] = v
# c_state_dict.update(c_model_dict)
# CA_Net.load_state_dict(c_state_dict)
# CA_Net.to(my_device)


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


# def pil_loader(path, rgb=True):
#     with open(path, "rb") as f:
#         with Image.open(f) as img:
#             if rgb:
#                 return img.convert("RGB")
#             else:
#                 return img.convert("F")
# my_transform = transforms.Compose(
#             [transforms.Resize((192, 640), interpolation=Image.ANTIALIAS),
#              transforms.ToTensor()])
#
#
# im_v = pil_loader('/home/dut616/project/monodepth2/Camera_0/rgb_00000.jpg')
#
# input_v = my_transform(im_v).unsqueeze(0)
#
# a1 = feature_ex(input_v.to(my_device))[("f_disp", 0)]
# # a1 = feature_ex.encoder(input_v.to(my_device))[1]
# b = a1[0].mean(0, True).squeeze(0)
# im1 = b.detach().cpu().numpy()
# plt.imshow(im1, vmin=-0.2, vmax=0.5, cmap=plt.cm.jet)
# plt.show()


for batch_idx, inputs in enumerate(val_loader):
    a1 = feature_ex(inputs[("color", 1, 0)].to(my_device))[("f_disp", 3)]
    a2 = feature_ex(inputs[("color", 0, 0)].to(my_device))[("f_disp", 3)]

    # a1 = feature_ex.encoder(inputs[("color", 1, 0)].to(my_device))[1]
    # a2 = feature_ex.encoder(inputs[("color", 0, 0)].to(my_device))[1]

    # a_1 = feature_ex.encoder(inputs[("color", 1, 0)].to(my_device))[-1]
    # a_0 = feature_ex.encoder(inputs[("color", 0, 0)].to(my_device))[-1]
    # a_11 = feature_ex.encoder(inputs[("color", -1, 0)].to(my_device))[-1]
    #
    # fea_ca = torch.cat((a_1, a_0), dim=1)
    # fea_ca = torch.cat((fea_ca, a_11), dim=1)
    #
    # channel_weights = CA_Net(fea_ca)
    #
    # a1 = a1 * channel_weights
    # a2 = a2 * channel_weights
    # print(channel_weights[7])





    im_show = 4
    # b2 = a2[4][30].squeeze(0)
    # b = a1[4][30].squeeze(0)
    b2 = a2[im_show].mean(0, True).squeeze(0)
    b = a1[im_show].mean(0, True).squeeze(0)
    # b2 = a2[im_show][6].squeeze(0)
    # b = a1[im_show][6].squeeze(0)

    im1 = b.detach().cpu().numpy()
    im2 = b2.detach().cpu().numpy()
    # im1 = (b.detach().cpu().numpy()-0.3)*100
    # im2 = (b2.detach().cpu().numpy()-0.3)*100

    # print(im1.max())

    im = inputs[("color", 1, 0)][im_show].cpu()
    im = im.permute(1, 2, 0).numpy()

    plt.subplot(1, 2, 1)
    # im2 = np.clip(im2, 0.2, 2)
    # _im=np.load('./111.npy')
    # print(_im.shape)
    plt.imshow(im1, vmin=-0.1, vmax=0.1, cmap=plt.cm.jet)
    # plt.subplot(1, 3, 2)
    # # plt.imshow(im2, vmin=-10, vmax=100, cmap=plt.cm.jet)
    # # plt.imshow(im1, vmin=-1, vmax=1, cmap=plt.cm.jet)
    # plt.imshow(im)
    plt.subplot(1, 2, 2)
    # plt.imsave('./111.jpg', im2, cmap='gray')
    plt.imshow(im2, vmin=-0.1, vmax=0.1, cmap=plt.cm.jet)
    # plt.imshow(im)
    plt.show()
    break

