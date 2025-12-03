# -*- coding: utf-8 -*-
# @Author  : Neil
# @Time    : 2024/5/24 15:10

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import numpy as np
import os
from utils.bicubic import BicubicDownSample
from tqdm import tqdm
import PIL
from PIL import Image
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from models.optimizer.ClampOptimizer import ClampOptimizer
from losses.blend_loss import BlendLossBuilder
from models.Net import Net
import cv2
from utils.data_utils import load_FS_latent
from utils.data_utils import cuda_unsqueeze
from utils.image_utils import load_img, dilate_erosion_mask_path, dilate_erosion_mask_tensor
from utils.model_utils import download_weight
toPIL = tv.transforms.ToPILImage()


class Blending(nn.Module):
    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        if not net:
            self.net = Net(self.opts)
        else:
            self.net = net
        self.load_downsampling()
        self.load_segmentation_network()
        self.build_blend_loss_builder()

    def load_downsampling(self):
        self.downsample = BicubicDownSample(factor=self.opts.size // 512)
        self.downsample256 = BicubicDownSample(factor=self.opts.size // 256)

    def load_segmentation_network(self):
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.opts.device)
        if not os.path.exists(self.opts.seg_ckpt):
            download_weight(self.opts.seg_ckpt)
        self.seg.load_state_dict(torch.load(self.opts.seg_ckpt))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()

    def build_blend_optimizer(self):
        interpolation_latent = torch.zeros((18, 512), requires_grad=True, device=self.opts.device)
        opt_blend = ClampOptimizer(torch.optim.Adam, [interpolation_latent], lr=self.opts.learning_rate)
        return opt_blend, interpolation_latent

    def build_blend_loss_builder(self):
        self.loss_builder = BlendLossBuilder(self.opts)

    def blend_images(self, im1_path, im2_path, im3_path, sign="realistic"):
        device = self.opts.device
        output_dir = self.opts.output_dir

        im_name_1 = os.path.splitext(os.path.basename(im1_path))[0]
        im_name_2 = os.path.splitext(os.path.basename(im2_path))[0]
        im_name_3 = os.path.splitext(os.path.basename(im3_path))[0]
        I_1 = load_img(im1_path, downsample=True).to(device).unsqueeze(0)
        I_3 = load_img(im3_path, downsample=True).to(device).unsqueeze(0)
        HM_1D, _ = cuda_unsqueeze(
            dilate_erosion_mask_path(im1_path, self.seg),
            device
        )
        HM_3D, HM_3E = cuda_unsqueeze(
            dilate_erosion_mask_path(im3_path, self.seg),
            device
        )
        opt_blend, interpolation_latent = self.build_blend_optimizer()
        latent_1, latent_F_mixed = load_FS_latent(
            os.path.join(
                output_dir,
                "Align_{}".format(sign),
                "{}_{}.npz".format(im_name_1, im_name_3)
            ), device
        )
        latent_3, _ = load_FS_latent(
            os.path.join(
                output_dir,
                "FS",
                "{}.npz".format(im_name_3)
            ), device
        )
        with torch.no_grad():
            I_X, _ = self.net.generator(
                [latent_1],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F_mixed
            )
            I_X_0_1 = (I_X + 1) / 2
            IM = (self.downsample(I_X_0_1) - seg_mean) / seg_std
            down_seg, _, _ = self.seg(IM)
            current_mask = torch.argmax(down_seg, dim=1).long().cpu().float()
            HM_X = torch.where(
                current_mask == 10,
                torch.ones_like(current_mask),
                torch.zeros_like(current_mask)
            )
            HM_X = F.interpolate(
                HM_X.unsqueeze(0),
                size=(256, 256),
                mode="nearest"
            ).squeeze()
            HM_XD, _ = cuda_unsqueeze(dilate_erosion_mask_tensor(HM_X), device)
            target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

        pbar = tqdm(range(self.opts.blend_steps), desc="Blend", leave=False)
        all_loss = 0
        iters=0
        for step in pbar:
            opt_blend.zero_grad()
            latent_mixed = latent_1 + interpolation_latent.unsqueeze(0) * (latent_3 - latent_1)
            I_G, _ = self.net.generator(
                [latent_mixed],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F_mixed
            )
            I_G_0_1 = (I_G + 1) / 2
            im_dict = {
                "gen_im": self.downsample256(I_G),
                "im_1": I_1,
                "im_3": I_3,
                "mask_face": target_mask,
                "mask_hair": HM_3E
            }
            loss, loss_dict = self.loss_builder(**im_dict)
            all_loss += loss
            iters += 1
            # if self.opts.verbose:
            #     pbar.set_description(
            #         'Blend Loss: {:.3f}, face: {:.3f}, hair: {:.3f}'.format(
            #             loss, loss_dict['face'], loss_dict['hair']
            #         )
            #     )
            # print(loss)
            loss.backward()
            opt_blend.step()

        # Load F code from  '{}_{}.npz'.format(im_name_1, im_name_2)
        _, latent_F_mixed = load_FS_latent(
            os.path.join(
                output_dir,
                "Align_{}".format(sign),
                "{}_{}.npz".format(im_name_1, im_name_2)
            ), device
        )
        I_G, _ = self.net.generator(
            [latent_mixed],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_F_mixed
        )
        self.save_blend_results(
            im_name_1, im_name_2, im_name_3, sign,
            I_G, latent_mixed, latent_F_mixed
        )
        return all_loss / iters

    def save_blend_results(
            self, im_name_1, im_name_2, im_name_3, sign,
            gen_im, latent_in, latent_F
    ):
        save_im = toPIL(
            (
                (gen_im[0] + 1) / 2
            ).detach().cpu().clamp(0, 1)
        )
        save_dir = os.path.join(
            self.opts.output_dir, "Blend_{}".format(sign)
        )
        os.makedirs(save_dir, exist_ok=True)
        latent_path = os.path.join(
            save_dir,
            "{}_{}_{}.npz".format(im_name_1, im_name_2, im_name_3)
        )
        img_path = os.path.join(
            save_dir,
            "{}_{}_{}.png".format(im_name_1, im_name_2, im_name_3)
        )
        output_img_path = os.path.join(
            self.opts.output_dir,
            "{}_{}_{}_{}.png".format(im_name_1, im_name_2, im_name_3, sign)
        )
        save_im.save(img_path)
        save_im.save(output_img_path)
        np.savez(
            latent_path,
            latent_in=latent_in.detach().cpu().numpy(),
            latent_F=latent_F.detach().cpu().numpy()
        )
