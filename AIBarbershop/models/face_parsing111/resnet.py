# -*- coding: utf-8 -*-
# @Author  : Neil
# @Time    : 2024/5/21 17:47
# state_dict = zoo.load_url(resnet18_url)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as zoo


resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


def Conv3x3(in_planes, out_planes, stride=1):
    # 3X3 conv with padding
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


def create_layer(in_chn, out_chn, b_num, stride=1):
    layers = [BasicBlock(in_chn, out_chn, stride=stride)]
    for i in range(b_num - 1):
        layers.append(BasicBlock(out_chn, out_chn, stride=stride))
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    def __init__(self, in_chn, out_chn, stride=1):
        super().__init__()
        self.conv1 = Conv3x3(in_chn, out_chn, stride)
        self.bn1 = nn.BatchNorm2d(out_chn)
        self.conv2 = Conv3x3(out_chn, out_chn)
        self.bn2 = nn.BatchNorm2d(out_chn)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chn != out_chn or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_chn, out_chn,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_chn),
            )

    def forward(self, X):
        res = self.conv1(X)
        res = F.relu(self.bn1(res))
        res = self.conv2(res)
        res = self.bn2(res)
        shortcut = X
        if self.downsample is not None:
            shortcut = self.downsample(X)
        out = res + shortcut
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer(64, 64, b_num=2, stride=1)
        self.layer2 = create_layer(64, 128, b_num=2, stride=2)
        self.layer3 = create_layer(128, 256, b_num=2, stride=2)
        self.layer4 = create_layer(256, 512, b_num=2, stride=2)
        self.init_weight()

    def forward(self, X):
        X = self.conv1(X)
        X = F.relu(self.bn1(X))
        X = self.maxpool(X)

        X = self.layer1(X)
        feat8 = self.layer2(X)          # 1/8
        feat16 = self.layer3(feat8)     # 1/16
        feat32 = self.layer4(feat16)    # 1/32
        return feat8, feat16, feat32

    def init_weight(self):
        # !!!!!!!!!!!!!!!!!!!!!
        state_dict = zoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if "fc" in k:
                continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_param, nowd_param = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_param.append(module.weight)
                if module.bias is not None:
                    nowd_param.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_param += list(module.parameters())
        return wd_param, nowd_param


# if __name__ == '__main__':
#     net = ResNet18()
#     X = torch.randn(16, 3, 224, 224)
#     out = net(X)
#     print(out[0].size())
#     print(out[1].size())
#     print(out[2].size())
#     net.get_params()
