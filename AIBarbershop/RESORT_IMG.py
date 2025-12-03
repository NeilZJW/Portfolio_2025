# -*- coding: utf-8 -*-
# @Author  : Neil
# @Time    : 2024/7/2 19:41

import cv2
import os

input_dir = "D:\\张峻玮\\4\\Summy\\7.3\\datasets"
count = 0
for img in os.listdir(input_dir):
    old_dir = os.path.join(input_dir, img)
    filename = os.path.splitext(img)[0]
    if filename != count:
        filetype = os.path.splitext(img)[1]
        new_dir = os.path.join(input_dir, "img" + str(count) + filetype)
        os.rename(old_dir, new_dir)
    count += 1





