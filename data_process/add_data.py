import pandas as pd
from collections import Counter
import pickle
import sys
import csv
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)
path = '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/zixun_tag/tag_extraction/data/'
idf_dic = pickle.load(open(path + 'idf_dic.txt', 'rb'))
wordpos_dic = pickle.load(open(path + 'wordpos_dic.txt', 'rb'))
stop_words = pickle.load(open(path + 'stop_words.txt', 'rb'))


def check_valid_wordpos(word):
    if len(wordpos_dic[word]) == 0:
        return True
    for v in wordpos_dic[word]:
        if 'n' in v or 'v' in v:
            return True
    return False


def check_format(check_str):
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


def check_word(word):
    """
    是否适合作为负样本的候选词
    :param word:
    :return:
    """
    if check_format(word) and check_valid_wordpos(word) and word not in stop_words and len(word) > 1:
        return True
    else:
        return False


def top_tfidf_word(title, content):
    """

    :param word: 候选词
    :param title: 标题
    :param content: 正文
    :param dic: idf词典
    :return: 特征集
    """
    tfidf_dic = {}
    # 候选词在正文出现频率
    corpus = content.split() + title.split()
    counter = dict(Counter(corpus))
    # 以50%概率选择标题中权重最大的词，另外50%概率选择正文中权重最大的词
    for word in set(corpus):
        if not check_word(word):
            continue
        tf = counter[word] if word in counter else 0

        # 候选词idf
        idf = idf_dic.get(word, 1.5)

        # 候选词tf-idf
        tfidf = tf * idf
        tfidf_dic[word] = tfidf

    words = [x[0] for x in sorted(tfidf_dic.items(), key=lambda x: x[1], reverse=True)]
    if len(words) == 0 or words[0] not in title or tfidf_dic[words[0]] < 10 or idf_dic.get(words[0], 1.5) < 2:
        return ''
    else:
        return words[0]


if __name__ == '__main__':
    print('read data...')
    path = '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/zixun_tag/tag_extraction/data/'
    data = pd.read_csv(path + 'toutiao_tag.csv', encoding='utf-8', engine='python')
    doc_tags = {}
    for docid, word in data[['docid', 'word']].values:
        if docid not in doc_tags:
            doc_tags[docid] = set()
        doc_tags[docid].add(word)
    dic = {'docid': [], 'title': [], 'content': [], 'word': []}
    doc_set = set()
    for docid, title, content in tqdm(data[['docid', 'title', 'content']].values):
        if docid in doc_set:
            continue
        add_word = top_tfidf_word(title, content)
        if len(add_word) > 1 and add_word not in doc_tags[docid]:
            dic['docid'].append(docid)
            dic['title'].append(title)
            dic['content'].append(content)
            dic['word'].append(add_word)
            doc_set.add(docid)
    add_data = pd.DataFrame(dic)
    add_data.to_csv('add_data.csv', encoding='utf-8', index=False)
