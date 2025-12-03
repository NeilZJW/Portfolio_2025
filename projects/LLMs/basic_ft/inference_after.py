#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DeepSeek 
@File    ：inference_after.py
@Author  ：Neil
@Date    ：2025/2/13 21:35 
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, InferArguments, infer_main, merge_lora
)
last_model_ckpt = "Output/checkpoint-62"
infer_args = InferArguments(
    model_type=ModelType.deepseek_r1_distill,
    ckpt_dir="Model/R1_Distil",
    adapters=last_model_ckpt,
    max_batch_size=256
)
merge_lora(infer_args, device_map="cpu")
# help(InferArguments)
result = infer_main(infer_args)