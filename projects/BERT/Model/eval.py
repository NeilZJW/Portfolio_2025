# -*- coding: utf-8 -*-
# @Author  : Neil
# @Time    : 2025/6/24 12:40

import pandas as pd
import torch
from DataPreprocessing.DataPreprocessing import DataPreprocessing
from transformers import BertForSequenceClassification, BertTokenizer


class ModelEvel:
    def __init__(self, path=r"E:\PyCharm 2020.2.3\BERT2025\Model\my_resume_classifier"):
        self.path = path


class ModelSlection(ModelEvel):
    def __init__(self):
        super().__init__()

    def bert_classification(self):
        model = BertForSequenceClassification.from_pretrained(self.path)
        tokenizer = BertTokenizer.from_pretrained(self.path)
        model.eval()
        id2label = model.config.id2label
        return model, tokenizer, id2label


class ClassificationTask:
    def __init__(self, model, tokenizer, id2label):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label

    def classify_text(self, text):
        try:
            device = self.model.device
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            pred_id = outputs.logits.argmax(dim=1).item()
            return self.id2label[pred_id]
        except Exception as e:
            return f"Error: {str(e)}"

    def classify_csv(self, file):
        pass


if __name__ == '__main__':
    model, tokenizer, id2label = ModelSlection().bert_classification()
    model.eval()
    example = "Proficient in Java and Python, built REST APIS with Flask."
    CT = ClassificationTask(model, tokenizer, id2label)
    result = CT.classify_text(example)
    print("Predicted Class: {}".format(result))
