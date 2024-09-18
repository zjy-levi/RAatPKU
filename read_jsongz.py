#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   read_jsongz.py
@Time    :   2024/09/18 09:39:40
@Author  :   _zJy_
@Version :   1.0
@License :   (C)Copyright 2024, MIT License
@Contact :   jinyuzh@hku.hk
@Desc    :   使用Goemotion数据集微调后的BERT模型，对数据集中的文本进行情感分析，提取其中的失望情感。在本代码中实现了批次处理，以提高处理速度。1. 自定义数据集类，继承自torch.utils.data.Dataset，用于加载数据集。2. 使用DataLoader加载数据集。3. 使用模型进行推理，提取失望情感的概率。4. 将结果保存到CSV文件中。
'''

import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import argparse
import gzip
from tqdm import tqdm

from transformers import BertTokenizer
from bert_finetune import CustomBERTModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# Define the labels
labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
def read_jsongz(file_name,num=1e6):
    fpth = os.path.join('data',file_name +'.jsonl.gz')
    with gzip.open(fpth, 'rt', encoding='utf-8') as f:
        # 逐行读取文件内容
        lines = []
        for i, line in enumerate(f):
            if i >= num:
                break
            lines.append(line)
    
    # 将读取的行转换为 DataFrame
    df = pd.read_json(''.join(lines), lines=True)
    return df
def add_config(config, extra_params):
    """向config中添加额外的参数

    Args:
        config (AutoModelConfig): 预训练模型的配置
        extra_params (dict): 需要额外传入的配置参数

    Returns:
        AutoModelConfig: 传入额外参数后的配置
    """
    for key, value in extra_params.items():
        if hasattr(config, key):
            print(f"Warning: Key '{key}' already exists in the config with value '{getattr(config, key)}'. It will be overwritten.")
        setattr(config, key, value)
    return config
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return inputs

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--fn', type=str)
    arg.add_argument('--device', default=0,type=int)
    args = arg.parse_args()
    # 合并title和content字段
    df = read_jsongz(args.fn)
    # df = read_jsongz('Appliances')
    if df.shape[0]>1e6:
        df = df.iloc[:int(1e6),:] # 只取前100万条数据
    print(df.columns)
    df['content'] = df['title'] + ' ' + df['text']
    print('Read file successfully')
    # Initialize the zero-shot classification pipeline
    model = CustomBERTModel.from_pretrained('model/fine_tuned_bert')
    model.cuda(args.device)
    tokenizer = BertTokenizer.from_pretrained("model/fine_tuned_bert")

    # 创建数据集和数据加载器
    dataset = TextDataset(df['content'].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    disappointment = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k:v.squeeze(1).cuda(args.device) for k,v in batch.items()}
            outputs = model(**batch)
            probs = outputs['logits'].softmax(dim=-1)
            disappointment.extend(probs[:,labels.index('disappointment')].tolist())
        # outputs = model(**tokenizer(df['content'].tolist(), truncation=True, padding='max_length', max_length=128, return_tensors="pt"))
    df['disappointment'] = disappointment
    df[['asin','parent_asin','user_id','disappointment']].to_csv(f'data/{args.fn}_disappointment.csv', index=False)
    # df[['asin','parent_asin','user_id']].to_csv(f'data/Appliances_disappointment.csv', index=False)
