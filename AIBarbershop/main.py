# -*- coding: utf-8 -*-
# author: Neil

import argparse
import torch
import numpy as np
import sys
import os
import dlib
from PIL import Image
from models.Embedding import Embedding
from models.Alignment import Alignment
from models.Blending import Blending


def main(args):
    ii2s = Embedding(args)
    im_ID_path = os.path.join(args.input_dir, args.im_I_path)
    im_S_path = os.path.join(args.input_dir, args.im_S_path)
    im_A_path = os.path.join(args.input_dir, args.im_A_path)
    im_set = {im_ID_path, im_S_path, im_A_path}
    ii2s.invert_img_in_W([*im_set])
    ii2s.invert_img_in_FS([*im_set])

    align = Alignment(args)
    align.align_images(
        im_ID_path, im_S_path, sign=args.sign,
        align_more_region=False,
        smooth=args.smooth
    )
    if im_S_path != im_A_path:
        align.align_images(
            im_ID_path, im_A_path, sign=args.sign,
            align_more_region=False,
            smooth=args.smooth,
            save_intermediate=False
        )

    blend = Blending(args)
    blend.blend_images(im_ID_path, im_S_path, im_A_path, sign=args.sign)


def add_parser(img_p1, img_p2, img_p3, output_name=None):
    output_name = "./output/" + output_name

    parser = argparse.ArgumentParser(description="AI-Barbershop")
    # I/O
    parser.add_argument("--input_dir", type=str, default="input/face", help="The directory of the images to be inverted")
    parser.add_argument("--output_dir", type=str, default=output_name, help="The directory to save the latent codes and inversion images")
    parser.add_argument('--im_I_path', type=str, default=img_p1, help='Identity image')
    parser.add_argument('--im_S_path', type=str, default=img_p2, help='Structure image')
    parser.add_argument('--im_A_path', type=str, default=img_p3, help='Appearance image')
    parser.add_argument("--sign", type=str, default="realistic", help='realistic or fidelity results')
    parser.add_argument("--smooth", type=int, default=5, help='dilation and erosion parameter')

    # StyleGANv2
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--ckpt", type=str, default="pretrained_models/ffhq.pt")
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)

    # Arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tile_latent", action="store_true", help="Whether to forcibly tile the same latent N times")
    parser.add_argument("--opt_name", type=str, default="adam", help="Optimizer to use in projected gradient descent")
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate to use during optimization')
    parser.add_argument("--verbose", action="store_true", help="Print loss information")
    parser.add_argument("--save_intermediate", action="store_true", help="Whether to store and save intermediate HR and LR images during optimization")
    parser.add_argument("--save_interval", type=int, default=300, help="Latent checkpoint interval")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seg_ckpt", type=str, default="pretrained_models/seg.pth", help="ckpt for segmentation net")

    # Embedding Loss Options
    # default 1100/250
    parser.add_argument("--percept_lambda", type=float, default=1.0, help="Perceptual loss multiplier factor")
    parser.add_argument("--l2_lambda", type=float, default=1.0, help="L2 loss multiplier factor")
    parser.add_argument("--p_norm_lambda", type=float, default=0.001, help="P-norm Regularizer multiplier factor")
    parser.add_argument("--l_F_lambda", type=float, default=0.1, help="L_F loss multiplier factor")
    parser.add_argument("--W_steps", type=int, default=400, help="Number of W space optimization steps")
    parser.add_argument("--FS_steps", type=int, default=100, help="Number of FS space optimization steps")

    # Aligning Loss Options
    parser.add_argument("--ce_lambda", type=float, default=1.0, help='cross entropy loss multiplier factor')
    parser.add_argument("--style_lambda", type=str, default=4e4, help='style loss multiplier factor')
    parser.add_argument("--align_steps1", type=int, default=80, help="")
    parser.add_argument("--align_steps2", type=int, default=50, help="")

    # Blending Loss Optiongs
    parser.add_argument('--face_lambda', type=float, default=1.0, help='')
    parser.add_argument('--hair_lambda', type=str, default=1.0, help='')
    parser.add_argument('--blend_steps', type=int, default=200, help='')

    # args = parser.parse_args()
    # return args
    return parser

#
# if __name__ == '__main__':
#     p_im1 = "ZRX.png"
#     p_im2 = "c4.png"
#     p_im3 = "s4.png"
#     args = add_parser(p_im1, p_im2, p_im3)
#     main(args)
