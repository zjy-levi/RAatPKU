import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

import os
import itertools
import argparse

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--reg', type=str, default=None, required=False)
    arg.add_argument('--fn', type=str, default=None, required=False, help='The file name of regression data')
    arg.add_argument('--formula', type=str, default=None, required=False, help='The formula of regression model')
    args = arg.parse_args()
    file_list = os.listdir('data')
    # 如果还没有合并
    if True in [s.__contains__('merged') for s in file_list]:# 读取文件夹中的所有文件,如果有merged文件则不再合并
        # print('Files have been merged')
        pass
    else:
        print('Files have not been merged!')
        print('Merging files...')
        disappointment_files = [f for f in file_list if f.__contains__('disappointment')]
        regression_files = [f for f in file_list if f.__contains__('regression')]
        pair = [(i,j) for i,j in itertools.product(disappointment_files, regression_files) if i.split('_')[0] == j.split('_')[0]]
        print('Merging successfully!')
    
    # 开始跑回归模型
    if args.reg == '1':
        # print('Running regression model...')
        # 读取文件
        fn = args.fn
        df = pd.read_csv(f'data/{fn}_merged.csv')
        model = smf.ols(formula=args.formula, data=df).fit()
        print(model.summary())
    elif args.reg == 'all':
        fn = args.fn
        df = pd.read_csv(f'data/{fn}_merged.csv')
        
        # Baseline model
        baseline_model = smf.ols(formula='np.log(helpful_vote+1) ~ disappointment', data=df).fit()
        print("Baseline model summary:")
        print(baseline_model.summary())
        
        # Sentiment model
        sentiment_model = smf.ols(formula='np.log(helpful_vote+1) ~ disappointment + sadness + anger + fear + disgust + surprise + joy', data=df).fit()
        print("Sentiment model summary:")
        print(sentiment_model.summary())
        
        # Disappointment model
        disappointment_model = smf.ols(formula='np.log(helpful_vote+1) ~ disappointment + rating + np.log(titleLen+1) + np.log(contentLen+1) + np.log(imageNum+1)', data=df).fit()
        print("Disappointment model summary:")
        print(disappointment_model.summary())
        
        # All vars model
        all_vars_model = smf.ols(formula='np.log(helpful_vote+1) ~ disappointment + sadness + anger + fear + disgust + surprise + joy + rating + np.log(titleLen+1) + np.log(contentLen+1) + np.log(imageNum+1)', data=df).fit()
        print("All vars model summary:")
        print(all_vars_model.summary())
        # df = pd.read_csv('data/merged_data.csv')
        # print('Read file successfully')
