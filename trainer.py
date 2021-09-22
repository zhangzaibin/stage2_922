# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

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
from manynetworks import ResnetEncoderMatching
from cvmap_networks import cvmap
from torch.autograd import Variable
from collections import OrderedDict
from IPython import embed
import matplotlib.pyplot as plt


####use the feature extractor form supervised model in v-kitti
my_device = torch.device("cuda:0")

mcnet = MCNet(11,5, 'linear').to(my_device)
feature_ex = mcnet.feature_extraction
# print(feature_ex)
# checkpoint = torch.load("./stage1_pretrained/dpsnet_2_checkpoint.pth.tar")['state_dict']
checkpoint = torch.load("./stage1_pretrained/stage1_gan/dpsnet_4_checkpoint.pth.tar")['state_dict']
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
#
#
# for p in CA_Net.parameters():
#     p.requires_grad = False





# mcnet = MCNet(50,1.5).to(my_device)
# feature_ex = mcnet.feature_extraction
# checkpoint = torch.load("./dpsnet_8_checkpoint.pth.tar")['state_dict']
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


# #####  use the costvolume in manydepth
# my_device = torch.device("cuda:0")
# fea_encoder = ResnetEncoderMatching(18, True, input_height=192, input_width=640, adaptive_bins=True, min_depth_bin=0.1,
#                                     max_depth_bin=20, depth_binning='linear', num_depth_bins=96)
# checkpoint = torch.load("./encoder.pth")
# new_state_dict = OrderedDict()
# for k, v in checkpoint.items():
#     if k in fea_encoder.state_dict().keys():
#         new_state_dict[k] = v
#
# fea_encoder.load_state_dict(new_state_dict)
# feature_ex = nn.Sequential(fea_encoder.layer0, fea_encoder.layer1)
# feature_ex.to(my_device)
#
# for p in feature_ex.parameters():
#     p.requires_grad = False




# my_device = torch.device("cuda:0")
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




_DEPTH_COLORMAP = plt.get_cmap('plasma', 100)  # for plotting

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.backproject_depth_p = {}
        self.project_3d = {}
        self.project_3d_p = {}
        for scale in self.opt.scales:
            h = 48 // (2 ** scale)
            w = 160 // (2 ** scale)
            h_p = self.opt.height // (2 ** scale)
            w_p = self.opt.width // (2 ** scale)

            self.backproject_depth_p[scale] = BackprojectDepth(self.opt.batch_size, h_p, w_p)
            self.backproject_depth_p[scale].to(self.device)

            self.project_3d_p[scale] = Project3D(self.opt.batch_size, h_p, w_p)
            self.project_3d_p[scale].to(self.device)
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)
        # self.backproject_depth = {}
        # self.project_3d = {}
        # for scale in self.opt.scales:
        #     h = self.opt.height // (2 ** scale)
        #     w = self.opt.width // (2 ** scale)
        #
        #     self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
        #     self.backproject_depth[scale].to(self.device)
        #
        #     self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
        #     self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_features_pred(inputs, outputs)
        # self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()


########
    def generate_features_pred(self, inputs, outputs):
        """Generate the warped (reprojected) features for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp_p = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                disp_f = F.interpolate(
                    disp, [48, 160], mode="bilinear", align_corners=True)
                source_scale = 0

            _, depth = disp_to_depth(disp_p, self.opt.min_depth, self.opt.max_depth)
            _f, depth_f = disp_to_depth(disp_f, self.opt.min_depth, self.opt.max_depth)
            # get the depth from the disp
            outputs[("depth", 0, scale)] = depth
            # resize the depth to the feature size
            # print(depth.shape)

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    # T = transformation_from_parameters(
                    #          axisangle[:, 0], translation[:, 0], frame_id < 0)
                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)


                # K = inputs[("inv_K", source_scale)].clone()
                # K[:, 0, :] /= 4.0
                # K[:, 1, :] /= 4.0
                #
                # inv_k = torch.zeros_like(K)
                # for i in range(inv_k.shape[0]):
                #     inv_k[i, :, :] = torch.pinverse(K[i, :, :])

                # cam_points_f = self.backproject_depth[source_scale](
                #     depth_f, inv_k)
                # pix_coords_f = self.project_3d[source_scale](
                #     cam_points_f, K, T)
                cam_points_f = self.backproject_depth[source_scale](
                    depth_f, inputs[("inv_K_f", source_scale)])
                pix_coords_f = self.project_3d[source_scale](
                    cam_points_f, inputs[("K_f", source_scale)], T)

                cam_points = self.backproject_depth_p[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d_p[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("sample_f", frame_id, scale)] = pix_coords_f

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    feature_ex(inputs[("color", frame_id, source_scale)])[("f_disp", scale)],
                    outputs[("sample_f", frame_id, scale)],
                    padding_mode="border", mode='bilinear')

                outputs[("color_p", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]


    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    # T = transformation_from_parameters(
                    #     axisangle[:, 0], translation[:, 0], frame_id < 0)
                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target, pred_p, target_p):
        abs_diff = torch.abs(pred - target)
        l1_loss = abs_diff.mean(1, True)

        #compute the photometric loss
        abs_diff_p = torch.abs(pred_p - target_p)
        l1_loss_p = abs_diff_p.mean(1, True)
        ssim_loss = self.ssim(pred_p, target_p).mean(1, True)
        reprojection_loss_p = 0.85 * ssim_loss + 0.15 * l1_loss_p
        return (l1_loss, reprojection_loss_p)
        # l1_loss = F.interpolate(
        #             l1_loss, [192, 640], mode="bilinear", align_corners=False)
        # print(l1_loss.shape)


        # cost = Variable(torch.FloatTensor(pred.size()[0], 1, 1, pred.size()[2],pred.size()[3]).zero_()).cuda()
        #     # cost[:, :pred.size()[1], 0, :, :] = target
        #     # cost[:, pred.size()[1]:, 0, :, :] = pred
        # cost[:, 0, 0, :, :] = torch.mean(torch.abs(target - pred), dim=1)
        #
        # costvolume = CVmap(cost, target)
        # costvolume = torch.squeeze(costvolume, 1)
        # abs_diff = 50.0 - costvolume
        # return abs_diff.mean(1, True)







    # def compute_reprojection_loss(self, pred, target):
    #     """Computes reprojection loss between a batch of predicted and target images
    #     """
    #     abs_diff = torch.abs(target - pred)
    #     l1_loss = abs_diff.mean(1, True)
    #
    #     if self.opt.no_ssim:
    #         reprojection_loss = l1_loss
    #     else:
    #         ssim_loss = self.ssim(pred, target).mean(1, True)
    #         reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
    #
         # return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        feature_loss_total=0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            feature_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = feature_ex(inputs[("color", 0, source_scale)])[("f_disp", scale)]
            target = F.interpolate(target, [48, 160], mode='bilinear')
            target_p = inputs[("color", 0, source_scale)]
            # target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                # pred = outputs[("color", frame_id, scale)]
                pred = F.interpolate(outputs[("color", frame_id, scale)], [48, 160], mode='bilinear')
                pred_p = outputs[("color_p", frame_id, scale)]
                feature_losses.append(self.compute_reprojection_loss(pred, target, pred_p, target_p)[0])
                reprojection_losses.append(self.compute_reprojection_loss(pred, target, pred_p, target_p)[1])
            # print(reprojection_losses)
            reprojection_losses = torch.cat(reprojection_losses, 1)
            feature_losses = torch.cat(feature_losses, 1)
            # reprojection_losses = torch.stack(reprojection_losses)


            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                # identity_feature_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = F.interpolate(feature_ex(inputs[("color", frame_id, source_scale)])[("f_disp", scale)], [48, 160], mode='bilinear')

                    pred_p = inputs[("color", frame_id, source_scale)]
                    # target = feature_ex(inputs[("color", frame_id, source_scale)])
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target, pred_p, target_p)[1])
                    # identity_feature_losses.append(
                    #     self.compute_reprojection_loss(pred, target, pred_p, target_p)[0])
                        # self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                # identity_feature_losses = torch.cat(identity_feature_losses, 1)
                # identity_reprojection_losses = torch.stack(identity_reprojection_losses)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                    # identity_feature_loss = identity_feature_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses
                    # identity_feature_loss = identity_feature_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)
                    # mask_f = F.interpolate(
                    #     mask, [48, 160],
                    #     mode="bilinear", align_corners=False)

                reprojection_losses *= mask
                # feature_losses *= mask_f

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                # feature_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses
                # feature_loss = feature_losses
            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001
                # identity_feature_loss += torch.randn(
                #     identity_feature_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
                # combined_f = torch.cat((identity_feature_loss, feature_loss), dim=1)
                # combined = torch.stack((identity_reprojection_loss, reprojection_loss))
            else:
                combined = reprojection_loss
                # combined_f = feature_loss

            # print(combined)
            if combined.shape[0] == 1:
                to_optimise = combined
                # to_optimise_f = combined_f
            else:
                # to_optimise, idxs = torch.min(combined, dim=0)
                to_optimise, idxs = torch.min(combined, dim=1)
                # to_optimise_f , idxs_f = torch.min(combined_f, dim=1)
            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()
                # outputs["identity_selection_f/{}".format(scale)] = (idxs_f > identity_feature_loss.shape[1] - 1).float()
                # outputs["fea_mask/{}".format(scale)] = _fea_mask
                _fea_mask = (idxs > identity_reprojection_loss.shape[1]).float()
                fea_mask = F.interpolate(_fea_mask.unsqueeze(1), [48, 160], mode="nearest")
                feature_losses, _ = torch.min(feature_losses, dim=1)
                _feature_losses = feature_losses.unsqueeze(1)
                feature_losses = _feature_losses * fea_mask


                feature_losses = feature_losses[:, :, 5:43, 5:155]
                fea_mask = fea_mask[:, :, 5:43, 5:155]
                # print(feature_losses.shape)
                outputs["fea_mask/{}".format(scale)] = (feature_losses).squeeze(1)
                outputs["fea_loss_visual/{}".format(scale)] = (_feature_losses).squeeze(1)
                outputs["photo_loss_visual/{}".format(scale)] = to_optimise.squeeze(1)

            loss += to_optimise.mean()
            loss += 0.1*(torch.sum(feature_losses) / torch.sum(fea_mask))
            #loss += 0.03 * torch.sum(feature_losses) / (torch.sum(fea_mask))
            #loss += 0.2*feature_losses.mean()
            feature_loss_total += 0.1*(torch.sum(feature_losses) / torch.sum(fea_mask))
            # feature_loss_total += 0.2*feature_losses.mean()
            # f_loss += 0.2*feature_losses.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        losses["feature_loss_total"] = feature_loss_total
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        writer.add_scalar("feature_loss_total", losses["feature_loss_total"].mean(), self.step)
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        fea_pred = colormap(outputs[("color", frame_id, s)][j].data.mean(0, True))
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            fea_pred, self.step)
                        fea_ori = colormap(feature_ex(inputs[("color", frame_id, s)])[("f_disp", 0)][j].data.mean(0, True))
                        writer.add_image(
                            "color_orig_{}_{}/{}".format(frame_id, s, j),
                            fea_ori, self.step)
                        writer.add_image(
                            "color_p_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color_p", frame_id, s)][j].data, self.step)
                        writer.add_image(
                            "feature_loss_visual".format(frame_id, s, j),
                            outputs["fea_loss_visual/{}".format(s)][j].data, self.step)
                        writer.add_image(
                            "photo_loss_visual".format(frame_id, s, j),
                            outputs["photo_loss_visual/{}".format(s)][j].data, self.step)



                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
                    # writer.add_image(
                    #     "automask_f_{}/{}".format(s, j),
                    #     outputs["identity_selection_f/{}".format(s)][j][None, ...], self.step)

                    writer.add_image(
                        "feamask_{}/{}".format(s, j),
                        outputs["fea_mask/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):

        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis