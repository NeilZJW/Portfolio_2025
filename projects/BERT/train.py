# -*- coding: utf-8 -*-
# @Author  : Neil
# @Time    : 2025/6/24 12:35

from trainer_build import BERTBuilder
from transformers import AutoTokenizer
from trainer_build import predict, plot_cm, plot_loss, acc_test, save_model


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    trainer = BERTBuilder().bert_trainer()
    trainer.train()
    _, _, tokenized_test = BERTBuilder().get_tokens()
    predictions = predict(trainer, tokenized_test)
    acc_test(predictions)
    plot_loss(trainer)
    plot_cm()
    save_model(trainer)
