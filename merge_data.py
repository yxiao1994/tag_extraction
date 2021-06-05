# coding:utf-8
# 合并正负样本

import pandas as pd
import re
from collections import Counter
import pickle
import math
import numpy as np
import sys
import csv

csv.field_size_limit(sys.maxsize)

if __name__ == '__main__':
    print('read data...')
    path ='/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/zixun_tag/tag_extraction/data/'
    pos = pd.read_csv(path + 'clean_positive_data.csv', encoding='utf-8', engine='python')
    pos['label'] = 1
    neg1 = pd.read_csv(path + 'neg_data_p1.csv', encoding='utf-8', engine='python')
    neg2 = pd.read_csv(path + 'neg_data_p2.csv', encoding='utf-8', engine='python')
    neg3 = pd.read_csv(path + 'neg_data_p3.csv', encoding='utf-8', engine='python')
    features = ['doc_id', 'title', 'content', 'word', 'label']
    data = pd.concat([pos[features], neg1[features], neg2[features], neg3[features]])
    data = data.drop_duplicates()
    data = data.dropna()
    data = data.sort_values(by='doc_id')
    print(data.shape)
    print('save data...')
    data.to_csv(path + 'data.csv', index=False, encoding='utf-8')