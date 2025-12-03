# -*- coding: utf-8 -*-
# @Author  : Neil
# @Time    : 2024/5/20 11:09


import torch
from torch import nn
from torch.nn import functional as F


class BicubicDownSample(nn.Module):
    def __init__(self, factor=4, cuda=True, padding='reflect'):
        super().__init__()
        self.factor = factor
        size = factor * 4
        k = torch.tensor(
            [
                self.bicubic_kernel(
                    i - torch.tensor(size / 2) + 0.5 / factor
                ) for i in range(size)
            ], dtype=torch.float32
        )
        k = k / torch.sum(k)
        k1 = torch.reshape(k, shape=(1, 1, size, 1))
        self.k1 = torch.cat([k1, k1, k1], dim=0)
        k2 = torch.reshape(k, shape=(1, 1, 1, size))
        self.k2 = torch.cat([k2, k2, k2], dim=0)
        self.cuda = ".cuda" if cuda else ""
        self.padding = padding
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, nhwc=False, clip_round=False, byte_output=False):
        filter_height = self.factor * 4
        filter_width = self.factor * 4
        stride = self.factor

        pad_along_height = max(filter_height - stride, 0)
        pad_along_width = max(filter_width - stride, 0)
        filter1 = self.k1.type("torch{}.FloatTensor".format(self.cuda))
        filter2 = self.k2.type("torch{}.FloatTensor".format(self.cuda))

        # compute actual padding values for each side
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # apply mirror padding
        if nhwc:
            # NHWC -> NCHW
            x = torch.transpose(
                torch.transpose(x, 2, 3), 1, 2
            )

        # downscaling performed by 1-d convolution
        x = F.pad(x, (0, 0, pad_top, pad_bottom), self.padding)
        x = F.conv2d(input=x, weight=filter1, stride=(stride, 1), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)
        x = F.pad(x, (pad_left, pad_right, 0, 0), self.padding)
        x = F.conv2d(input=x, weight=filter2, stride=(1, stride), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)
        if nhwc:
            x = torch.transpose(
                torch.transpose(x, 1, 2), 1, 2
            )
        if byte_output:
            return x.type("torch.ByteTensor".format(self.cuda))
        else:
            return x

    def bicubic_kernel(self, x, a=-0.50):
        """
        This equation is exactly copied from the website below:
        https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic
        """
        x_abs = torch.abs(x)
        if x_abs <= 1:
            return (a + 2.) * torch.pow(x_abs, 3.) - (a + 3.) * torch.pow(
                x_abs, 2
            ) + 1
        elif 1 < x_abs < 2:
            return a * torch.pow(x_abs, 3.) - 5. * a * torch.pow(
                x_abs, 2.
            ) + 8. * a * x_abs - 4. * a
        else:
            return 0.0
