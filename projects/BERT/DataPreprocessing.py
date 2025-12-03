# -*- coding: utf-8 -*-
# @Author  : Neil
# @Time    : 2025/6/24 11:08

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset


class DataLoading:
    def __init__(self, root=r"E:\PyCharm 2020.2.3\BERT2025\DataPreprocessing\Preprocessed_Data.csv"):
        self.data = pd.read_csv(root).drop_duplicates(subset=["Text"]).\
            reset_index(drop=True)
        self.label2id = dict()
        self.id2label = dict()


class DataPreprocessing(DataLoading):
    def __init__(self):
        super().__init__()

    def encode(self):
        le = LabelEncoder()
        self.data["GroudTruth"] = le.fit_transform(self.data["Category"])
        # print(le.classes_)
        self.label2id = {
            str(class_name): int(class_id) for class_name, class_id in zip(
                le.classes_, le.transform(le.classes_)
            )
        }
        # print(self.label2id)
        self.id2label = {v: k for k, v in self.label2id.items()}
        # print(self.id2label)
        return self.data, self.label2id, self.id2label

    def create_data_for_train(self, n_sample=40):
        df, label2id, id2label = self.encode()
        test_d = df.groupby("Category").sample(n=n_sample, random_state=42)
        # print(test_d)
        remaining_df = df[~df.index.isin(test_d.index)]
        train_d, val_d = train_test_split(
            remaining_df,
            test_size=0.2,
            stratify=remaining_df["Category"],
            random_state=42
        )
        train_df = train_d[["Text", "GroudTruth"]].rename(columns={
            "Text": "text",
            "GroudTruth": "label"
        })
        val_df = val_d[["Text", "GroudTruth"]].rename(columns={
            "Text": "text",
            "GroudTruth": "label"
        })
        test_df = test_d[["Text", "GroudTruth"]].rename(columns={
            "Text": "text",
            "GroudTruth": "label"
        })
        # print(test_df)
        train_dataset = Dataset.from_pandas(train_df).remove_columns(["__index_level_0__"])
        val_dataset = Dataset.from_pandas(val_df).remove_columns(["__index_level_0__"])
        test_dataset = Dataset.from_pandas(test_df).remove_columns(["__index_level_0__"])
        # print(test_dataset)
        return test_dataset, val_dataset, train_dataset, label2id, id2label


if __name__ == '__main__':
    # data = DataPreprocessing().data
    # print(data)
    DataPreprocessing().create_data_for_train()

