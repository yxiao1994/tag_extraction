# coding:utf-8
import pandas as pd
import re
from collections import Counter
import pickle
import math
import numpy as np
import sys
import csv

csv.field_size_limit(sys.maxsize)

FILE_NUM = 350000


def get_features(word, title, content, dic):
    """

    :param word: 候选词
    :param title: 标题
    :param content: 正文
    :param dic: idf词典
    :return: 特征集
    """
    if len(content) == 0:
        content = '文本 没有 内容'
    if len(title) == 0:
        title = '标题 没有 内容'
    # 候选词是否出现在标题中
    word_in_title = 1 if word in title.split('|||') else 0

    # 候选词是否在书名号内
    x = content.replace('|||', '')  # 去空格
    f = re.findall(u'《(.*?)》', x)
    is_keyword = 1 if word in f else 0

    # 候选词在正文出现频率
    counter = dict(Counter(content.split('|||')))
    tf = counter[word] if word in counter else 0

    # 候选词idf
    idf = 1.5
    if word in dic and dic[word] >= 3:
        idf = math.log10(FILE_NUM * 1.0 / dic[word])

    # 候选词tf-idf
    tfidf = tf * idf

    # 候选词在标题出现频率
    counter = dict(Counter(title.split('|||')))
    tf2 = counter[word] if word in counter else 0
    tfidf2 = tf2 * idf

    # 候选词在文本中出现位置
    words = content.split('|||')
    position_in_content = (words.index(word) + 1) if word in words else len(words)
    position_rate_in_content = 1.0 * position_in_content / (len(words) if len(words) > 0 else 1)
    is_first_content_word = (position_in_content == 1)

    # 候选词在第几个句子中出现
    position_in_sentence = -1
    sentences = re.split(u'。|！|\!|\.|？|\?', content)
    # print('///'.join(sentences))
    if word in words:
        for j, sentence in enumerate(sentences):
            if word in sentence:
                # print(word + str(i))
                position_in_sentence = j + 1
                break
    position_rate_in_sentence = 1.1 if (word not in words) else (1.0 * position_in_sentence / len(sentences))
    is_first_sentence_word = (position_in_sentence == 1)

    return [word_in_title, is_keyword, tf, idf, tfidf, tf2, tfidf2,
            position_in_content, position_rate_in_content, is_first_content_word,
            position_in_sentence, position_rate_in_sentence, is_first_sentence_word]


def feature_eng(df, dic):
    print('get features...')
    i = 0
    features = []
    for word, title, content in df[['word', 'title', 'content']].values:
        i += 1
        if i % 10000 == 0:
            print("{} data processed".format(i))
        l = get_features(word, title, content, dic)
        features.append(l)
    features = np.array(features)
    return features


if __name__ == '__main__':
    print('read data...')
    path = '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/zixun_tag/tag_extraction/data/'
    idf_dic = pickle.load(open(path + 'idf_dic.txt', 'rb'))
    train = pd.read_csv(path + 'train.csv', encoding='utf-8', engine='python')
    train['content'] = train['content'].fillna('文本 没有 内容')
    train['title'] = train['title'].fillna('标题 没有 内容')
    train = train.dropna()
    train = train.reset_index(drop=True)

    print(train.shape)
    features = feature_eng(train, idf_dic)
    print('save data...')
    np.save(path + 'train_features', features)
