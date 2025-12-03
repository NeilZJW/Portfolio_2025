# -*- coding: utf-8 -*-
# @Author  : Neil
# @Time    : 2024/5/24 15:15

import math
import torch
from torch.optim import Optimizer
import numpy as np


class ClampOptimizer(Optimizer):
    def __init__(self, optimizer, params, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        for param in self.params:
            tem_latent_norm = torch.clamp(param.data, 0, 1)
            param.data.add_(tem_latent_norm - param.data)
        return loss

    def zero_grad(self):
        self.opt.zero_grad()
