# coding:utf-8
# qq切词的使用

import pandas as pd
import sys
from collections import defaultdict
import codecs
import csv
import pickle
from tqdm import tqdm
import jieba.posseg as pseg

csv.field_size_limit(sys.maxsize)


if __name__ == '__main__':
    print('read data...')
    path = '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/zixun_tag/tag_extraction/data/'
    data = pd.read_csv(path + 'toutiao_tag.csv', encoding='utf-8', engine='python')
    print(data.shape)
    data = data[['title', 'content', 'docid']]
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    data = data.dropna()
    print(data.shape)
    #data = data.sample(frac=0.7)

    wordpos_dic = defaultdict(set)
    fo = codecs.open(path + 'word_pos.txt', 'w', 'utf-8')
    for title, content in tqdm(data[['title', 'content']].values):
        title = title.replace(' ', '')
        content = content.replace(' ', '')
        res1 = pseg.cut(title)
        for t in res1:
            wordpos_dic[t.word].add(t.flag)
        res2 = pseg.cut(content)
        for t in res2:
            wordpos_dic[t.word].add(t.flag)
    pickle.dump(wordpos_dic, open(path + 'wordpos_dic.txt', 'wb'))