#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DeepSeek 
@File    ：1.py
@Author  ：Neil
@Date    ：2025/3/1 0:58 
"""
import requests

response = requests.get(
    "https://www.modelscope.cn/api/v1/datasets/swift/self-cognition/repo?Source=SDK&Revision=master&FilePath=self_cognition.jsonl",
    verify=False
)
print(response.text)

