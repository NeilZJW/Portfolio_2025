#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import fasttext
import os
from data_loader import data_get




# 数据预处理模块
def data_preprocess(df, column_name, train=True):
    if train:
        temp_path = "fasttext_train.txt"
    else:
        temp_path = "fasttext_test.txt"

    with open(temp_path, 'w', encoding='utf-8') as f:
        for desc in df[column_name]:
            f.write(desc + '\n')

    print(f"数据处理完成，保存为 {temp_path}")
    return temp_path

# 训练模型
def train_model(train_file):
    model = fasttext.train_supervised(
        input=train_file,
        epoch=100,
        lr=0.1,
        wordNgrams=2,
        loss="ova",
        verbose=2
    )
    model.save_model("resume_classifier.bin")
    print("模型训练完成，已保存为 resume_classifier.bin")
    return model

# 评估模型
def evaluate_model(test_file, df_test, model_path="resume_classifier.bin"):
    model = fasttext.load_model(model_path)

    predictions = []
    correct = 0

    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line, true_label in zip(lines, df_test['GroundTruth']):
        line = line.strip()
        pred_label, prob = model.predict(line, k=1)
        predicted_label = pred_label[0].replace('__label__', '')
        predictions.append(predicted_label)

        if predicted_label == true_label:
            correct += 1

        print(f"Описание: {line}")
        print(f"Предсказанный класс: {predicted_label}, Вероятность: {prob[0]:.2f}, 真实类别: {true_label}\n")

    accuracy = correct / len(df_test)
    print(f"整体准确率: {accuracy:.2%}")

    df_test['Predicted'] = predictions
    df_test.to_csv("test_with_predictions.csv", index=False)
    print("预测结果已保存到 test_with_predictions.csv")

def eval_in_app(test_file, df_test, model_path="resume_classifier.bin"):
    model = fasttext.load_model(model_path)

    predictions = []

    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        pred_label, prob = model.predict(line, k=1)
        predicted_label = pred_label[0].replace('__label__', '')
        predictions.append(predicted_label)
    df_test['Predicted'] = predictions
    df_test.to_csv("test_with_predictions.csv", index=False)
    return "Successful!"


# 主程序
if __name__ == '__main__':
    df_final_trainset = data_get("train.csv")
    df_train_distinct = data_get("test.csv")

    # 数据预处理
    train_file = data_preprocess(df_final_trainset, "Описание_FastText", train=True)
    test_file = data_preprocess(df_train_distinct, "Описание", train=False)

    eval_in_app(test_file, df_train_distinct)

    # # 模型训练
    # train_model(train_file)
    #
    # # 模型评估
    # evaluate_model(test_file, df_train_distinct)
