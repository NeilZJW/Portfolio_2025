#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DeepSeek 
@File    ：get_model.py
@Author  ：Neil
@Date    ：2025/2/13 15:32 
"""
#模型下载
from modelscope import snapshot_download
# model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', local_dir="Model/R1_Distil")
# help(snapshot_download)

model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', local_dir="Model/R1_Distil/32B")
