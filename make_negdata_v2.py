# coding:utf-8
import pandas as pd
import random
from collections import defaultdict
import pickle
import sys
import csv
import numpy as np

csv.field_size_limit(sys.maxsize)


def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))


def is_neg_word(word, tags, embedding):
    try:
        for keyword in tags:
            if (keyword == word) or (keyword in word) or (word in keyword):
                return False
            if (word in embedding) and (keyword in embedding) and (
                    cosine_similarity(embedding[word], embedding[keyword]) > 0.4):
                # print(word.encode('utf8') + ' similar to ' + keyword.encode('utf-8'))
                return False
        return True
    except:
        return False


path = '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/zixun_tag/tag_extraction/data/'
embedding_index = pickle.load(open(path + 'embeddings_index.txt', 'rb'))
print('read data...')
df = pd.read_csv(path + 'clean_positive_data.csv', encoding='utf-8', engine='python')

corpus_tag_dic = defaultdict(list)
for word, doc_id in df[['word', 'doc_id']].values:
    corpus_tag_dic[doc_id].append(word)

pos_tag = list(df['word'])

df = df[['title', 'content', 'doc_id']]
df = df.drop_duplicates()
df = df.reset_index(drop=True)
df = df.dropna()

dic = {
    'title': [],
    'word': [],
    'content': [],
    'label': [],
    'doc_id': []
}
print('generate negative samples...')
i = 0
for title, content, doc_id in df[['title', 'content', 'doc_id']].values:
    i += 1
    if i % 10000 == 0:
        print("{} data processed".format(i))
        #break
    word_set = set(content.split('|||') + title.split('|||'))
    sample_neg = random.sample(pos_tag, 2)
    # 抽取在pos_tag但是不在文本的词语作为负样本
    for word in sample_neg:
        # 候选关键词筛选
        if (word not in word_set) and (is_neg_word(word, corpus_tag_dic[doc_id], embedding_index)):
            dic['title'].append(title)
            dic['word'].append(word)
            dic['content'].append(content)
            dic['label'].append(0)
            dic['doc_id'].append(doc_id)

neg_data = pd.DataFrame(dic)
neg_data = neg_data.drop_duplicates()
neg_data.to_csv(path + 'neg_data_p2.csv', encoding='utf-8', index=False)

