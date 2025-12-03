#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：HR 
@File    ：rule.py
@Author  ：Neil
@Date    ：2025/2/27 17:55 
"""

import regex as re
import pandas as pd
from data_loader import data_get


def filter_by_gender(data, gender_pattern):
    # муж or жен
    pattern = fr"^{gender_pattern}.*"
    return data[data["Пол"].str.match(pattern, re.IGNORECASE, na=False)]

def filter_by_age(data, age_pattern):
    # T: 18-
    # A - Молодежь: 18-25
    # B - Молодые специалисты: 26-35
    # C - Опытные специалисты: 36-45
    # D - Старшие специалисты: 46+
    if age_pattern == "T":
        return data[data["Возраст"] < 18]

    elif age_pattern == "A":
        return data[(data["Возраст"] >= 18) & (data["Возраст"] <= 25)]

    elif age_pattern == "B":
        return data[(data["Возраст"] >= 26) & (data["Возраст"] <= 35)]

    elif age_pattern == "C":
        return data[(data["Возраст"] >= 36) & (data["Возраст"] <= 45)]

    elif age_pattern == "D":
        return data[data["Возраст"] >= 46]


def filter_by_education(data, education_pattern):
    # кандидат магистр бакалавр доктор
    pattern = fr"^\b{education_pattern}\b"
    return data[data["Образование"].str.contains(pattern, re.IGNORECASE, na=False)]

def filter_by_major(data, major_pattern):
    return data[data["Predicted"] == major_pattern]

if __name__ == '__main__':
    data = data_get('test.csv')
    male_data = filter_by_gender(data, "Муж")
    print(male_data["Пол"])
    # male_data.to_csv("1.csv", index=False)
    # age_data = filter_by_age(data, "A")
#
#     education_data = filter_by_education("магистр")
#     # кандидат магистр бакалавр доктор
#     print(education_data["Образование"])
