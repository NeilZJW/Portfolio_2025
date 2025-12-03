#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SBZRX 
@File    ：Modelhub-GPT4o.py
@Author  ：Neil
@Date    ：2025/4/9 17:09 
"""
from opencompass.models import OpenAI

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

models = [
    dict(
        abbr='Modelhub-GPT4o',
        type=OpenAI,
        path='https://modelhub.puyuan.tech/api/v1',  # 你的API地址
        key='3036378489:uChLp1kRYpi5wh7FQ1L41qstPvE40cju',  # 直接写用户名:密码
        model='gpt-4o',   # 指定使用的模型
        meta_template=api_meta_template,
        query_per_second=3,
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=1
    )
]
