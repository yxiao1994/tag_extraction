# coding:utf-8
# qq切词的使用

import pandas as pd
import sys
from collections import defaultdict
import codecs
sys.path.append('/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/qqseg_new')
import qq_seg_multi_thread
qq_seg_multi_thread.InitQQSegEnv()
import csv
import pickle
from tqdm import tqdm
csv.field_size_limit(sys.maxsize)


if __name__ == '__main__':
    print('read data...')
    path = '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/zixun_tag/tag_extraction/data/'
    data = pd.read_csv(path + 'clean_origin_data.csv', encoding='utf-8', engine='python')
    data = data.dropna()
    data['content'] = data['content'].fillna(u'文本 没有 内容')
    data['title'] = data['title'].fillna(u'标题 没有 内容')

    wordpos_dic = defaultdict(set)
    fo = codecs.open(path + 'word_pos.txt', 'w', 'utf-8')
    for title, content in tqdm(data[['title', 'content']].values):
        title = title.replace('|||', '')
        content = content.replace('|||', '')
        res1 = qq_seg_multi_thread.cut_mix(title, with_pos=True)
        for t in res1:
            wordpos_dic[t[0]].add(t[1])
        res2 = qq_seg_multi_thread.cut_mix(content, with_pos=True)
        for t in res2:
            wordpos_dic[t[0]].add(t[1])
    for key in wordpos_dic:
        fo.write(key + '\t' + '|'.join(wordpos_dic[key]) + '\n')
    fo.close()
    pickle.dump(wordpos_dic, open(path + 'wordpos_dic.txt', 'wb'))
