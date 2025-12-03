#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：HR 
@File    ：data_loader.py
@Author  ：Neil
@Date    ：2025/2/27 17:56 
"""

import os
import pandas as pd


def data_get(path):
    # # 如果你之前保存了数据集为CSV，可以用以下方法加载：
    # df_final_trainset = pd.read_csv("train.csv")
    # df_train_distinct = pd.read_csv("test.csv")
    # # 数据导出
    # df_final_trainset.to_csv("train.csv", index=False, quoting=1)
    # df_train_distinct.to_csv("test.csv", index=False, quoting=1)
    if path == "test_with_predictions.csv":
        if not os.path.exists("test_with_predictions.csv"):
            return pd.DataFrame()
        else:
            data = pd.read_csv(path, encoding="utf-8")
            try:
                d_data = data.drop("GroundTruth", axis=1)
                return d_data
            except:
                return data
    else:
        data = pd.read_csv(path, encoding="utf-8")
        return data


if __name__ == '__main__':
    data = data_get('russian_resume_dataset.csv')
    print(data.head())

