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
import tqdm

my_device = torch.device("cuda:0")

mcnet = MCNet(50,1.5).to(my_device)
feature_ex = mcnet.feature_extraction
# print(feature_ex)
checkpoint = torch.load("./stage1_pretrained/new_gaussion/dpsnet_3_checkpoint.pth.tar")['state_dict']

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
    opts.frame_ids, 4, is_train=False, img_ext=img_ext)
train_loader = DataLoader(
    train_dataset, 12, True,
    num_workers=11, pin_memory=True, drop_last=True)
val_dataset = dataset(
    opts.data_path, val_filenames, opts.height, opts.width,
    opts.frame_ids, 4, is_train=False, img_ext=img_ext)
val_loader = DataLoader(
    val_dataset, 12, shuffle=False,
    num_workers=11, pin_memory=True, drop_last=True)
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


for batch_idx, inputs in tqdm.tqdm(enumerate(val_loader)):
    a1 = feature_ex(inputs[("color", 1, 0)].to(my_device))[("f_disp", 0)]
    a2 = feature_ex(inputs[("color", 0, 0)].to(my_device))[("f_disp", 0)]
    a_1 = feature_ex(inputs[("color", -1, 0)].to(my_device))[("f_disp", 0)]
    # a1 = feature_ex.encoder(inputs[("color", 1, 0)].to(my_device))[1]
    # a2 = feature_ex.encoder(inputs[("color", 0, 0)].to(my_device))[1]



    for k in range(12):
        b0 = a2[k].mean(0, True).squeeze(0)
        b1 = a1[k].mean(0, True).squeeze(0)
        b_1 = a_1[k].mean(0, True).squeeze(0)
        im0 = b0.detach().cpu().numpy()
        im1 = b1.detach().cpu().numpy()
        im_1 = b_1.detach().cpu().numpy()
        # print(os.path.join(inputs[("color_path", 0, -1)][0].split('/kitti_data/')[0], 'feature_map',
        #              inputs[("color_path", 0, -1)][0].split('/kitti_data/')[1].split('.jpg')[0]+'.npy'))
        # print(inputs[("color_path", 0, -1)][0].split('kitti_data')[0])
        # print(os.path.join(inputs[("color_path", 0, -1)][0].split('/kitti_data/')[0], 'feature_map', inputs[("color_path", 0, -1)][0].split('/kitti_data/')[1]))

        np.save(
            os.path.join(inputs[("color_path", 0, -1)][k].split('/kitti_data/')[0], 'feature_map', inputs[("color_path", 0, -1)][k].split('/kitti_data/')[1].split('.jpg')[0]+'.npy'), im0
        )
        np.save(
            os.path.join(inputs[("color_path", 1, -1)][k].split('/kitti_data/')[0], 'feature_map',
                         inputs[("color_path", 1, -1)][k].split('/kitti_data/')[1].split('.jpg')[0]+'.npy'), im1
        )
        np.save(
            os.path.join(inputs[("color_path", -1, -1)][k].split('/kitti_data/')[0], 'feature_map',
                         inputs[("color_path", -1, -1)][k].split('/kitti_data/')[1].split('.jpg')[0]+'.npy'), im_1
        )




    # plt.subplot(1, 2, 1)
    # # im2 = np.clip(im2, 0.2, 2)
    # plt.imshow(im1, vmin=-0.2, vmax=0.5, cmap=plt.cm.jet)
    # # plt.subplot(1, 3, 2)
    # # # plt.imshow(im2, vmin=-10, vmax=100, cmap=plt.cm.jet)
    # # # plt.imshow(im1, vmin=-1, vmax=1, cmap=plt.cm.jet)
    # # plt.imshow(im)
    # plt.subplot(1, 2, 2)
    # plt.imshow(im2, vmin=-0.2, vmax=0.5, cmap=plt.cm.jet)
    # # plt.imshow(im)
    # plt.show()
    # break

