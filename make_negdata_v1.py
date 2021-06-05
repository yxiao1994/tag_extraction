# coding:utf-8
# 对关键词抽取训练集构造负样本
import pandas as pd
import pickle
import random
import codecs
import sys
import csv
from tqdm import tqdm
from collections import Counter, defaultdict
import numpy as np

csv.field_size_limit(sys.maxsize)


def check_valid_wordpos(word, wordpos_dic):
    if 'n' in wordpos_dic[word] or 'v' in wordpos_dic[word]:
        return True
    else:
        return False


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


def cosine_similarity(word1, word2, embedding_dic, sim_dic):
    """
    计算词向量之间的相似度
    :param word1: 词语1
    :param word2: 词语2
    :param embedding: 词向量字典
    :param sim_dic: 用于缓存已经计算过的相似度值
    :return:
    """
    if word1 > word2:
        word1, word2 = word2, word1
    s = word1 + '|' + word2
    if s not in sim_dic:
        vector1 = embedding_dic[word1]
        vector2 = embedding_dic[word2]
        res = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
        sim_dic[s] = res
        return res
    else:
        # print('recorded!')
        return sim_dic[s]


def is_good_word(word, tags, embedding, threshold, sim_dic):
    """
    根据候选词和标题/正标签的相似度，判断词语是否适合作为负样本
    :param word: 候选词
    :param tags: 关键词
    :param embedding: 词向量的字典
    :param threshold: 相似度阈值。大于该阈值不适合
    :param sim_dic:
    :return:
    """
    for keyword in tags:
        if keyword in word or word in keyword:
            return True
        if (word in embedding) and (keyword in embedding) and \
                cosine_similarity(word, keyword, embedding, sim_dic) > threshold:
            # print(word.encode('utf8') + ' similar to ' + keyword.encode('utf-8'))
            return True
    return False


if __name__ == '__main__':
    # 记录词向量相似度的词典
    sim_dic = {}

    print('read data...')
    path = '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/zixun_tag/tag_extraction/data/'

    # 记录词性的词典
    wordpos_dic = pickle.load(open(path + 'wordpos_dic.txt', 'rb'))

    # embedding字典
    print('read embedding matrix...')
    embeddings_index = pickle.load(open(path + 'embeddings_index.txt', 'rb'))
    print('word embedding', len(embeddings_index))
    print(cosine_similarity('王者荣耀', '和平精英', embeddings_index, sim_dic))

    # 停用词集合
    stop_words = set()
    f = codecs.open(path + 'stop_words.txt', 'r', 'utf-8').readlines()
    for line in f:
        stop_words.add(line.strip('\n'))

    # 词语的idf词典
    idf_dic = pickle.load(open(path + 'idf_dic.txt', 'rb'))
    print('read data...')

    # 读取正样本，格式为docid/title/content/tag/1
    df = pd.read_csv(path + 'clean_positive_data.csv', encoding='utf-8', engine='python')

    # 记录每条docid对应的关键词（正样本）集合
    corpus_tag_dic = defaultdict(list)
    for word, doc_id in df[['word', 'doc_id']].values:
        corpus_tag_dic[doc_id].append(word)

    # 正样本（关键词）集合
    pos_tag_set = set(df['word'])

    # 正样本（关键词）的列表，由于列表包含了正样本的出现次数，后续会根据出现次数采样
    pos_tag_list = list(df['word'])

    # 记录正样本中关键词出现次数，构造负样本会根据该次数采样
    pos_word_count = dict(Counter(pos_tag_list))

    # 控制词语作为负样本的次数
    record_negword = defaultdict(int)

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

    for title, content, doc_id in tqdm(df[['title', 'content', 'doc_id']].values):
        # 抽取在html但是不在pos_tag的词语作为负样本，主要是一些无意义的词
        neg_candidates = set()
        words = set(content.split('|||'))
        for word in (words - pos_tag_set):
            # 候选关键词筛选，名词、不是正样本、字数大于1
            if (len(word) > 1) & (check_valid_wordpos(word, wordpos_dic)) & (word not in corpus_tag_dic[doc_id]) & (
                    word not in stop_words) & (check_valid(word)):
                neg_candidates.add(word)

        # 从候选词随机抽取5个负样本
        sample_neg = random.sample(neg_candidates, min(2, len(neg_candidates)))
        for word in sample_neg:
            record_negword[word] += 1
            dic['title'].append(title)
            dic['word'].append(word)
            dic['content'].append(content)
            dic['label'].append(0)
            dic['doc_id'].append(doc_id)

        # 第二部分，困难样本构造，词语出现在语料和正样本关键词集合中
        # 通过计算和标题词语相似度，低于某个阈值的认为是负样本
        neg_candidates = set()
        for word in (pos_tag_set & words):
            # 候选关键词筛选，名词、不是正样本、字数大于1i
            try:
                if (word not in corpus_tag_dic[doc_id]) & (record_negword[word] < 1000):
                    keywords = title.split('|||') + corpus_tag_dic[doc_id]
                    if not is_good_word(word, keywords, embeddings_index, 0.4, sim_dic):
                        neg_candidates.add(word)
            except:
                pass
        # 从候选词随机抽取5个负样本,按照正样本中的关键词出现频率采样
        neg_can = []
        for w in neg_candidates:
            for j in range(pos_word_count[w]):
                neg_can.append(w)
        sample_neg = random.sample(neg_can, min(5, len(neg_candidates)))
        for word in sample_neg:
            record_negword[word] += 1
            dic['title'].append(title)
            dic['word'].append(word)
            dic['content'].append(content)
            dic['label'].append(0)
            dic['doc_id'].append(doc_id)

        # 第三部分，对标题中的词语构造负样本
        neg_candidates = set()
        for word in set(title.split('|||')) - pos_tag_set:
            # 候选关键词筛选，名词、不是正样本、字数大于1
            try:
                if (word not in corpus_tag_dic[doc_id]) & check_valid(word):
                    if not is_good_word(word, corpus_tag_dic[doc_id], embeddings_index, 0.2, sim_dic):
                        neg_candidates.add(word)
            except:
                pass
        # 从候选词随机抽取5个负样本
        sample_neg = random.sample(neg_candidates, min(2, len(neg_candidates)))
        for word in sample_neg:
            record_negword[word] += 1
            dic['title'].append(title)
            dic['word'].append(word)
            dic['content'].append(content)
            dic['label'].append(0)
            dic['doc_id'].append(doc_id)

    neg_data = pd.DataFrame(dic)
    neg_data = neg_data.drop_duplicates()
    neg_data.to_csv(path + 'neg_data_p1.csv', encoding='utf-8', index=False)
