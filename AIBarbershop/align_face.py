# -*- coding: utf-8 -*-
# @Author  : Neil
# @Time    : 2024/5/30 14:56

import dlib
from pathlib import Path
import argparse
import torchvision as tv
from utils.shape_predictor import align_face
from PIL import Image


parser = argparse.ArgumentParser(description="Align_face")
parser.add_argument("-unprocessed_dir", type=str, default="unprocessed", help="directory with unprocessed images")
parser.add_argument("-output_dir", type=str, default="input/face", help="output directory")
parser.add_argument("-output_size", type=int, default=1024, help='size to downscale the input images to, must be power of 2')
parser.add_argument("-seed", type=int, help="manual seed to use")
parser.add_argument("-cache_dir", type=str, default="cache", help="cache directory for model weights")
parser.add_argument("-inter_method", type=str, default="bicubic")
args = parser.parse_args()

cache_dir = Path(args.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
print("Loading Shape Predictor!")
f = "cache/b3e61b9e2b05a24cd93cf529b98917c7_shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(f)

for im in Path(args.unprocessed_dir).glob("*.*"):
    faces = align_face(str(im), predictor)
    for i, face in enumerate(faces):
        if args.output_size:
            factor = 1024 // args.output_size
            assert args.output_size * factor == 1024
            face_tensor = tv.transforms.ToTensor()(face).unsqueeze(0).cuda()
            face_tensor_lr = face_tensor[0].cpu().detach().clamp(0, 1)
            face = tv.transforms.ToPILImage()(face_tensor_lr)
            if factor != 1:
                face = face.resize(
                    (args.output_size, args.output_size),
                    Image.LANCZOS
                )
        if len(faces) > 1:
            face.save(Path(args.output_dir) / (im.stem + f"_{i}.png"))
        else:
            face.save(Path(args.output_dir) / (im.stem + ".png"))
