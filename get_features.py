# coding:utf-8
import re
from collections import Counter
import math
import sys
import csv
import numpy as np

csv.field_size_limit(sys.maxsize)


def get_features(word, title, content, dic):
    """

    :param word: 候选词
    :param title: 标题
    :param content: 正文
    :param dic: idf词典
    :return: 特征集
    """
    if content is None:
        content = ''
    if title is None:
        title = ''
    # 候选词是否出现在标题中
    word_in_title = 1 if word in title.split() else 0

    # 候选词是否在书名号内
    x = content.replace(' ', '')  # 去空格
    f = re.findall('《(.*?)》', x)
    is_keyword = 1 if word in f else 0

    # 候选词在正文出现频率
    counter = dict(Counter(content.split()))
    tf = counter[word] if word in counter else 0

    # 候选词idf
    idf = dic.get(word, 1.5)

    # 候选词tf-idf
    tfidf = tf * idf

    # 候选词在标题出现频率
    counter = dict(Counter(title.split()))
    tf2 = counter[word] if word in counter else 0
    tfidf2 = tf2 * idf

    # 候选词在文本中出现位置
    words = content.split()
    position_in_content = (words.index(word) + 1) if word in words else len(words)
    position_rate_in_content = 1.0 * position_in_content / (len(words) if len(words) > 0 else 1)
    is_first_content_word = (position_in_content == 1)

    # 候选词在第几个句子中出现
    sentences = re.split('。|！|\!|\.|？|\?', content)
    position_in_sentence = len(sentences) + 1
    # print('///'.join(sentences))
    if word in words:
        for j, sentence in enumerate(sentences):
            if word in sentence:
                # print(word + str(i))
                position_in_sentence = j + 1
                break
    position_rate_in_sentence = 1.0 * position_in_sentence / len(sentences) \
        if len(sentences) > 0 else 0
    is_first_sentence_word = (position_in_sentence == 1)
    tf, idf, tfidf, tf2, tfidf2, position_in_content, position_in_sentence = \
        np.log1p([tf, idf, tfidf, tf2, tfidf2, position_in_content, position_in_sentence])
    return [word_in_title, is_keyword, tf, idf, tfidf, tf2, tfidf2,
            position_in_content, position_rate_in_content, is_first_content_word,
            position_in_sentence, position_rate_in_sentence, is_first_sentence_word]
