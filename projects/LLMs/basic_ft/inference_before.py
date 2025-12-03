#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DeepSeek 
@File    ：inference_before.py
@Author  ：Neil
@Date    ：2025/2/13 15:40 
"""
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：QWEN 
@File    ：inference_before.py
@Author  ：Neil
@Date    ：2024/11/15 2:12 
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from swift.llm import (
    ModelType, InferArguments, infer_main
)
infer_args = InferArguments(
    model_type=ModelType.deepseek_r1_distill,
    ckpt_dir="Model/R1_Distil/32B",
    max_batch_size=256
)
# help(InferArguments)
infer_main(infer_args)












# from swift.llm import (
#     get_model_tokenizer, get_template, inference, ModelType,
#     get_default_template_type, inference_stream
# )
from swift.utils import seed_everything
import torch
#
# model_type = ModelType.qwen_7b_chat
# template_type = get_default_template_type(model_type)
# print(f'template_type: {template_type}')
#
#
# kwargs = {}
# model_id_or_path = "llm_model_base/qwen/Qwen-7B-Chat"
# model, tokenizer = get_model_tokenizer(
#     model_type, torch.float16,
#     model_id_or_path=model_id_or_path,
#     model_kwargs={'device_map': 'cuda:0'},
#     **kwargs
# )
# # 修改max_new_tokens
# model.generation_config.max_new_tokens = 128
#
# template = get_template(template_type, tokenizer)
# seed_everything(42)
#
# query = '你是谁？'
# response, history = inference(model, template, query)
# print(f'response: {response}')
# print(f'history: {history}')
