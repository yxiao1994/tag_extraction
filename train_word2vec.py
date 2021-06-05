# coding:utf-8
# 训练词向量并保存为字典格式
import csv
import sys
csv.field_size_limit(sys.maxsize)
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import codecs
import pickle


if __name__ == '__main__':
    path = '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/zixun_tag/tag_extraction/data/'
    train = pd.read_csv(path + 'clean_origin_data.csv', encoding='utf-8', engine='python')
    train = train[['title', 'content']]
    train = train.drop_duplicates()
    train = train.fillna(u'没有 内容')

    content_corpus = [_.split('|||') for _ in train['content'].values]
    title_corpus = [_.split('|||') for _ in train['title'].values]
    seg_corpus = content_corpus + title_corpus

    print(len(seg_corpus))
    print('train word2vec...')
    model = Word2Vec(seg_corpus, size=200, min_count=5, sg=1, hs=0, negative=10, workers=500, window=5)
    model.wv.save_word2vec_format(path + "word_embedding.txt")
    print(' '.join([x[0].encode('utf-8') for x in (model.most_similar(u'医生'))]))
    print(' '.join([x[0].encode('utf-8') for x in (model.most_similar(u'教育'))]))
    print(' '.join([x[0].encode('utf-8') for x in (model.most_similar(u'妻子'))]))
    print(' '.join([x[0].encode('utf-8') for x in (model.most_similar(u'习近平'))]))
    print(' '.join([x[0].encode('utf-8') for x in (model.most_similar(u'汽车'))]))
    print(' '.join([x[0].encode('utf-8') for x in (model.most_similar(u'手机'))]))
    print(' '.join([x[0].encode('utf-8') for x in (model.most_similar(u'湖南卫视'))]))
    print(' '.join([x[0].encode('utf-8') for x in (model.most_similar(u'游戏'))]))

    embeddings_index = {}
    with codecs.open(path + 'word_embedding.txt', 'r', 'utf-8') as f:
        for i in f:
            try:
                values = i.split(' ')
                if len(values) != 201:
                    continue
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float')
                embeddings_index[word] = embedding
            except:
                print(i.encode('utf-8'))
    pickle.dump(embeddings_index, open(path + 'embeddings_index.txt', 'wb'))
