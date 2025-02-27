#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   zero-shot.py
@Time    :   2024/11/27 23:16:26
@Author  :   _zJy_
@Version :   1.0
@License :   (C)Copyright 2024, MIT License
@Contact :   jinyuzh@hku.hk
@Desc    :   Tater, T., Frassinelli, D., & im Walde, S. S. (2022, November). Concreteness vs. abstractness: A selectional preference perspective. In Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing: Student Research Workshop (pp. 92-98).
'''

from transformers import pipeline
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7,8,9'

import argparse
import gzip
from tqdm import tqdm
tqdm.pandas()
datadir = '/mnt/data_disk/chu123/jyuzh/TangChuang/data/'
# 设置缓存目录
os.environ['TRANSFORMERS_CACHE'] = '~/jyuzh/Practice/TangChuang/·model/.cache'
def read_jsongz(file_name,num=1e6):
    fpth = os.path.join(datadir,file_name +'.jsonl.gz')
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
def cal_abstract(text,pipe):
    if text=='' or text is None:
        return 0
        # 处理多条语句
    results = pipe(
        text,
        candidate_labels=["concrete", "abstract"],
        hypothesis_template="This review in Amazon is {}."
    )
    return results['scores'][1]

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--fn', type=str)
    arg.add_argument('--device', default=0,type=int)
    args = arg.parse_args()
    # 合并title和content字段
    df = read_jsongz(args.fn)
    if df.shape[0]>1e6:
        df = df.iloc[:int(1e6),:] # 只取前100万条数据
    print('Read file successfully')
    pipe = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", cache_dir='~/jyuzh/Practice/TangChuang/·model/.cache',device=args.device)    

    df['abstract'] = df['text'].progress_apply(lambda x: cal_abstract(x,pipe))
    df[['asin','parent_asin','user_id','abstract']].to_csv(os.path.join(datadir,f'{args.fn}_abstract.csv'), index=False)
    # print(df['abstract'])