# coding:utf-8
import pandas as pd
import pickle
import random
import codecs
import sys
import csv
from collections import Counter, defaultdict
import numpy as np
import math

csv.field_size_limit(sys.maxsize)
FILE_NUM = 390000


def check_valid(check_str):
    """
    检查词语是否为中文
    :param check_str: 输入词
    :return:true if the input str is chinese
    """
    all_digits = True
    for ch in check_str:
        is_chn = (ch >= '\u4e00') and (ch <= '\u9fa5')
        is_digit = (ch >= '\u0030') and (ch <= '\u0039')
        is_en = (ch >= '\u0041' and ch <= '\u005a') or (ch >= '\u0061' and ch <= '\u007a')
        if not is_chn and not is_digit and not is_en and ch != '-':
            return False
        if not is_digit:
            all_digits = False
    if all_digits:
        return False
    return True

def cosine_similarity(word1, word2, embedding, sim_dic):
    word1 = str(word1)
    word2 = str(word2)
    if (word1 in word2) or (word2 in word1):
        return 1
    if word1 > word2:
        word1, word2 = word2, word1
    s = word1 + '|' + word2
    if (word1 not in embedding) or (word2 not in embedding):
        # print(word1 + ',' + word2)
        return 1
    if s not in sim_dic:
        vector1 = embedding[word1]
        vector2 = embedding[word2]
        res = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
        sim_dic[s] = res
        return res
    else:
        # print('recorded!')
        return sim_dic[s]


def top_tfidf10_word(content, dic, keywords, embedding):
    """

    :param word: 候选词
    :param title: 标题
    :param content: 正文
    :param dic: idf词典
    :return: 特征集
    """
    tfidf_dic = {}
    # 候选词在正文出现频率
    counter = dict(Counter(content.split('|||')))
    res = []
    words = []
    for word in set(content.split('|||')):
        if not check_valid(word):
            continue
        tf = counter[word] if word in counter else 0

        # 候选词idf
        idf = 1.5
        if word in dic and dic[word] > 2:
            idf = math.log10(FILE_NUM * 1.0 / dic[word])
       
        # 候选词tf-idf
        tfidf = tf * idf
        if tfidf > 5:
            words.append(word)
    for word in words:
        flag = 1
        for keyword in keywords:
            similarity = cosine_similarity(word, keyword, embedding, sim_dic)
            # print(word + ',' + keyword + 'similarity:' + str(similarity))
            if similarity > 0.3:
                flag = 0
                break
        if flag == 1:
            res.append(word)

    return res


sim_dic = {}
print('read data...')
path = '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/zixun_tag/tag_extraction/data/'
print('read embedding matrix...')
embedding_index = pickle.load(open(path + 'embeddings_index.txt', 'rb'))
print(cosine_similarity('国足', '中国足球', embedding_index, sim_dic))
print(cosine_similarity('王者荣耀', '茶壶', embedding_index, sim_dic))
print(cosine_similarity('国足', '中国足球', embedding_index, sim_dic))
print(cosine_similarity('王者荣耀', '茶壶', embedding_index, sim_dic))

idf_dic = pickle.load(open(path + 'idf_dic.txt', 'rb'))
print('read data...')
df = pd.read_csv(path + 'clean_positive_data.csv', encoding='utf-8', engine='python')

corpus_tag_dic = defaultdict(list)
for word, doc_id in df[['word', 'doc_id']].values:
    corpus_tag_dic[doc_id].append(word)

df = df[['title', 'content', 'doc_id']]
df = df.drop_duplicates()
df = df.reset_index(drop=True)
df = df.dropna()
print('generate negative samples...')
dic = {
    'title': [],
    'word': [],
    'content': [],
    'label': [],
    'doc_id': []
}

i = 0
for title, content, doc_id in df[['title', 'content', 'doc_id']].values:
    i += 1
    if i % 10000 == 0:
        print("{} data processed".format(i))
       # break
    keywords = title.split('|||') + corpus_tag_dic[doc_id]
    for word in top_tfidf10_word(content, idf_dic, keywords, embedding_index):
        dic['title'].append(title)
        dic['word'].append(word)
        dic['content'].append(content)
        dic['label'].append(0)
        dic['doc_id'].append(doc_id)

neg_data = pd.DataFrame(dic)
neg_data = neg_data.drop_duplicates()
neg_data.to_csv(path + 'neg_data_p3.csv', encoding='utf-8', index=False)

