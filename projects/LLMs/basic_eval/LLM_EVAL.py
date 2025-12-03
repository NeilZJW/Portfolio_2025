#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SBZRX 
@File    ：LLM_EVAL.py
@Author  ：Neil
@Date    ：2025/4/9 16:32 
"""

from mmengine.config import read_base
from opencompass.models import OpenAI
with read_base():
    from opencompass.configs.models.ZRX.GPT_4o import models as gpt4o
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets


datasets = gsm8k_datasets
models = gpt4o

