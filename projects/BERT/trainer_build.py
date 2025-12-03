# -*- coding: utf-8 -*-
# @Author  : Neil
# @Time    : 2025/6/24 11:39

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
from DataPreprocessing.DataPreprocessing import DataPreprocessing
from transformers import BertForSequenceClassification
from transformers import TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from transformers import Trainer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("TkAgg")


class DataPrepare:
    def __init__(self):
        (
            self.test_dataset, self.val_dataset, self.train_dataset,
            self.label2id, self.id2label
        ) = DataPreprocessing().create_data_for_train()


class BERTBuilder(DataPrepare):
    def __init__(self):
        super().__init__()

    def get_tokens(self):
        tokenized_train = self.train_dataset.map(tokenizing, batched=True)
        tokenized_val = self.val_dataset.map(tokenizing, batched=True)
        tokenized_test = self.test_dataset.map(tokenizing, batched=True)
        return tokenized_train, tokenized_val, tokenized_test

    def bert_model(self):
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(self.label2id),
            label2id=self.label2id,
            id2label=self.id2label
        )
        return model

    def bert_arguments(self):
        training_args = TrainingArguments(
            output_dir="results",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="logs",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )
        return training_args

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        print(classification_report(labels, preds))
        return {"accuracy": acc}

    def bert_trainer(self):
        model = self.bert_model()
        training_args = self.bert_arguments()
        tokenized_train, tokenized_val, tokenized_test = self.get_tokens()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics
        )
        return trainer


def tokenizing(examples):
    return tokenizer(
      examples["text"],
      padding="max_length",
      truncation=True
    )


def predict(trainer, tokenized_examples):
    return trainer.predict(tokenized_examples)


def plot_loss(trainer):
    logs = pd.DataFrame(trainer.state.log_history)
    plt.figure(figsize=(10, 5))
    plt.plot(
        logs["step"],
        logs["loss"],
        label="Training Loss",
        marker="o"
    )
    if "eval_loss" in logs.columns:
        plt.plot(
            logs["step"],
            logs["eval_loss"],
            label="Validation Loss",
            marker="o"
        )
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def acc_test(predictions):
    global logits, labels, preds
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    print("Accuracy: {:.4f}".format(acc))


def plot_cm():
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(16, 16))
    id2label = DataPrepare().id2label
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=id2label.values(),
        yticklabels=id2label.values()
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.show()


def save_model(trainer, save_path="BERT/my_resume_classifier"):
    trainer.save_model(save_path)


if __name__ == '__main__':
    trainer = BERTBuilder().bert_trainer()
    _, _, tokenized_test = BERTBuilder().get_tokens()
    predictions = predict(trainer, tokenized_test)
    acc_test(predictions)
    plot_loss(trainer)
    plot_cm()
