# -*- coding: utf-8 -*-
# @Author  : Neil
# @Time    : 2024/5/2 15:31

import torch
from torch import nn
import numpy as np
import os
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
import PIL
import torchvision
from models.Net import Net
from utils.bicubic import BicubicDownSample
from utils.data_utils import convert_npy_code
from datasets.image_dataset import ImagesDatasets
from losses.embedding_loss import EmbeddingLossBuilder
toPIL = torchvision.transforms.ToPILImage()


class Embedding(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.net = Net(self.opts)
        self.load_downsampling()
        self.build_embedding_loss_builder()

    def load_downsampling(self):
        factor = self.opts.size // 256
        self.downsample = BicubicDownSample(factor=factor)

    def build_embedding_loss_builder(self):
        self.loss_builder = EmbeddingLossBuilder(self.opts)

    def build_W_optimizer(self):
        opt_dict ={
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "sgdm": partial(torch.optim.SGD, momentum=0.9),
            "adamax": torch.optim.Adamax
        }
        latent = []
        if self.opts.tile_latent:
            tmp = self.net.latent_avg.clone().detach().cuda()
            tmp.requires_grad = True
            for i in range(self.net.layer_num):
                latent.append(tmp)
            opt_W = opt_dict[self.opts.opt_name](
                [tmp], lr=self.opts.learning_rate
            )
        else:
            for i in range(self.net.layer_num):
                tmp = self.net.latent_avg.clone().detach().cuda()
                tmp.requires_grad = True
                latent.append(tmp)
            opt_W = opt_dict[self.opts.opt_name](
                latent, lr=self.opts.learning_rate
            )
        return opt_W, latent

    def build_FS_optimizer(self, latent_W, F_init):
        latent_F = F_init.clone().detach().requires_grad_(True)
        latent_S = []
        opt_dict = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "sgdm": partial(torch.optim.SGD, momentum=0.9),
            "adamax": torch.optim.Adamax
        }
        for i in range(self.net.layer_num):
            tmp = latent_W[0, i].clone()
            if i < self.net.S_index:
                tmp.requires_grad = False
            else:
                tmp.requires_grad = True
            latent_S.append(tmp)
        opt_FS = opt_dict[self.opts.opt_name](
            (latent_S[self.net.S_index:] + [latent_F]),
            lr=self.opts.learning_rate
        )
        return opt_FS, latent_F, latent_S

    def build_dataloader(self, image_path=None):
        self.dataset = ImagesDatasets(opts=self.opts, image_path=image_path)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        print("Number of images: {}".format(len(self.dataset)))

    def invert_img_in_W(self, image_path=None):
        self.build_dataloader(image_path=image_path)
        device = self.opts.device
        ibar = tqdm(self.dataloader, desc="Images")
        all_loss = 0
        iters = 0
        for ref_im_H, ref_im_L, ref_name in ibar:
            opt_W, latent = self.build_W_optimizer()
            pbar = tqdm(
                range(self.opts.W_steps), desc="Embedding", leave=False
            )
            for step in pbar:
                opt_W.zero_grad()
                latent_in = torch.stack(latent).unsqueeze(0)
                gen_im, _ = self.net.generator(
                    [latent_in], input_is_latent=True, return_latents=False
                )
                im_dict = {
                    "ref_im_H": ref_im_H.to(device),
                    "ref_im_L": ref_im_L.to(device),
                    "gen_im_H": gen_im,
                    "gen_im_L": self.downsample(gen_im)
                }
                loss, loss_dic = self.cal_loss(im_dict, latent_in)
                loss.backward()
                all_loss += loss
                iters += 1
                opt_W.step()

                if self.opts.verbose:
                    pbar.set_description(
                        "Embedding: Loss: {:.3f}, L2: {:.3f}, Perceptual Loss: {:.3f}, P-norm Loss: {:.3f}"
                        .format(loss, loss_dic["l2"], loss_dic["percep"], loss_dic["p-norm"])
                    )
                if self.opts.save_intermediate and step % self.opts.save_interval == 0:
                    self.save_W_intermediate_results(ref_name, gen_im, latent_in, step)
            self.save_W_results(ref_name, gen_im, latent_in)
        return all_loss / iters

    def invert_img_in_FS(self, image_path=None):
        self.build_dataloader(image_path=image_path)
        output_dir = self.opts.output_dir
        device = self.opts.device
        ibar = tqdm(self.dataloader, desc="images")
        all_loss = 0
        iters = 0

        for ref_im_H, ref_im_L, ref_name in ibar:
            latent_W_path = os.path.join(output_dir, "W+", "{}.npy".format(ref_name[0]))
            latent_W = torch.from_numpy(convert_npy_code(np.load(latent_W_path))).to(device)
            F_init, _ = self.net.generator(
                [latent_W], input_is_latent=True, return_latents=False,
                start_layer=0, end_layer=3
            )
            opt_FS, latent_F, latent_S = self.build_FS_optimizer(latent_W, F_init)
            pbar = tqdm(
                range(self.opts.FS_steps), desc="Embedding", leave=False
            )
            for step in pbar:
                opt_FS.zero_grad()
                latent_in = torch.stack(latent_S).unsqueeze(0)
                gen_im, _ = self.net.generator(
                    [latent_in], input_is_latent=True, return_latents=False,
                    start_layer=4, end_layer=8, layer_in=latent_F
                )
                im_dict = {
                    "ref_im_H": ref_im_H.to(device),
                    "ref_im_L": ref_im_L.to(device),
                    "gen_im_H": gen_im,
                    "gen_im_L": self.downsample(gen_im)
                }
                loss, loss_dict = self.cal_loss(im_dict, latent_in)
                loss.backward()
                all_loss += loss
                iters += 1
                opt_FS.step()

                if self.opts.verbose:
                    pbar.set_description(
                        "Embedding: Loss: {:.3f}, L2 Loss: {.3f}, Perceptual Loss: {:.3f}, P-norm Loss: {:.3f}, L_F Loss: {.3f}"
                        .format(loss, loss_dict["l2"], loss_dict["percep"], loss_dict["p-norm"], loss_dict["l_F"])
                    )
            self.save_FS_results(ref_name, gen_im, latent_in, latent_F)

        return all_loss / iters

    def cal_loss(self, im_dict, latent_in, latent_F=None, F_init=None):
        loss, loss_dict = self.loss_builder(**im_dict)
        p_norm_loss = self.net.cal_p_norm_loss(latent_in)
        loss_dict["p-norm"] = p_norm_loss
        loss += p_norm_loss
        if latent_F is not None and F_init is not None:
            l_F = self.net.cal_l_F(latent_F, F_init)
            loss_dict["l_F"] = l_F
            loss += l_F
        return loss, loss_dict

    def save_W_intermediate_results(self, ref_name, gen_im, latent_in, step):
        save_im = toPIL(
            ((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1)
        )
        save_latent = latent_in.detach().cpu().numpy()
        intermediate_folder = os.path.join(self.opts.output_dir, "W+", ref_name[0])
        os.makedirs(intermediate_folder, exist_ok=True)
        latent_path = os.path.join(
            intermediate_folder, "{}_{:04}.npy".format(ref_name[0], step)
        )
        img_path = os.path.join(
            intermediate_folder, "{}_{:04}.png".format(ref_name[0], step)
        )
        save_im.save(img_path)
        np.save(latent_path, save_latent)

    def save_W_results(self, ref_name, gen_im, latent_in):
        save_im = toPIL(
            ((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1)
        )
        save_latent = latent_in.detach().cpu().numpy()
        output_dir = os.path.join(self.opts.output_dir, "W+")
        os.makedirs(output_dir, exist_ok=True)
        latent_path = os.path.join(
            output_dir, "{}.npy".format(ref_name[0])
        )
        image_path = os.path.join(
            output_dir, "{}.png".format(ref_name[0])
        )
        save_im.save(image_path)
        np.save(latent_path, save_latent)

    def save_FS_results(self, ref_name, gen_im, latent_in, latent_F):
        save_im = toPIL(
            ((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1)
        )
        output_dir = os.path.join(self.opts.output_dir, "FS")
        os.makedirs(output_dir, exist_ok=True)
        latent_path = os.path.join(output_dir, "{}.npz".format(ref_name[0]))
        img_path = os.path.join(output_dir, "{}.png".format(ref_name[0]))
        save_im.save(img_path)
        np.savez(
            latent_path,
            latent_in=latent_in.detach().cpu().numpy(),
            latent_F=latent_F.detach().cpu().numpy()
        )

    def set_seed(self):
        if self.opts.seed:
            torch.manual_seed(self.opts.seed)
            torch.cuda.manual_seed(self.opts.seed)
            torch.backends.cudnn.deterministic = True
