# -*- coding: utf-8 -*-
# @Author  : Neil
# @Time    : 2024/5/22 18:08

import numpy as np
import os
from PIL import Image


def vis_seg(pred):
    num_label = 16
    color = np.array([
        [0, 0, 0],              # 0
        [102, 204, 255],        # 1
        [255, 204, 255],        # 2
        [255, 255, 153],        # 3
        [255, 255, 153],        # 4
        [255, 255, 102],        # 5
        [51, 255, 51],          # 6
        [0, 153, 255],          # 7
        [0, 255, 255],          # 8
        [0, 255, 255],          # 9
        [204, 102, 255],        # 10
        [0, 153, 255],          # 11
        [0, 255, 153],          # 12
        [0, 51, 0],
        [102, 153, 255],        # 14
        [255, 153, 102],        # 15
    ])
    h, w = np.shape(pred)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(num_label):
        mask = pred == i
        rgb[mask, None] = color[i, :]
    # current unk
    unk = pred == 255
    rgb[unk, None] = color[0, :]
    return rgb


def save_vis_mask(img_path1, img_path2, sign, output_dir, mask):
    im_name1 = os.path.splitext(os.path.basename(img_path1))[0]
    im_name2 = os.path.splitext(os.path.basename(img_path2))[0]
    vis_path = os.path.join(output_dir, "vis_mask_{}_{}_{}.png".format(
        im_name1, im_name2, sign
    ))
    vis_mask = vis_seg(mask)
    Image.fromarray(vis_mask).save(vis_path)
