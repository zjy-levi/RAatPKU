#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   inference_roberta_goemotion.py
@Time    :   2024/09/19 21:14:37
@Author  :   _zJy_
@Version :   1.0
@License :   (C)Copyright 2024, MIT License
@Contact :   jinyuzh@hku.hk
@Desc    :   使用Goemotion数据集微调后的RoBERTa模型，对数据集中的文本进行情感分析，提取其中的失望情感。在本代码中实现了批次处理，以提高处理速度。1. 自定义数据集类，继承自torch.utils.data.Dataset，用于加载数据集。2. 使用DataLoader加载数据集。3. 使用模型进行推理，提取失望情感的概率。4. 将结果保存到CSV文件中。
'''
import pandas as pd
import os
import argparse
from jsonpath import jsonpath
import json
from tqdm import tqdm

import transformers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, pipeline

class TextDataset(Dataset):
    def __init__(self, texts: list):
        self.texts = texts
        # self.tokenizer = tokenizer
        # self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return text

if __name__ == "__main__":

    classifier = pipeline(task="text-classification", model="roberta-base-go_emotions", top_k=None, device=0)
    # text这里只需要传入想分析disappointment的文本即可，下面是测试案例
    text = TextDataset(["I am so disappointed in you.", "I am so happy for you."])
    dataloader = DataLoader(text, batch_size=32, shuffle=False)
    disappointments = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            model_outputs = classifier(batch)
            model_outputs = json.loads(json.dumps(model_outputs, indent=4))
            disappointment = jsonpath(model_outputs, '$..[?(@.label=="disappointment")].score')
            disappointments.extend(disappointment)
            break
    print(disappointments)
