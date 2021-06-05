# coding:utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
import pickle
import sys
import csv

csv.field_size_limit(sys.maxsize)

MAX_NB_WORDS = 200000
MAX_NB_KEY_WORDS = 200000
EMBEDDING_DIM = 200
path = '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/zixun_tag/tag_extraction/data/'

if __name__ == '__main__':
    print('read data...')
    train = pd.read_csv(path + 'train.csv', encoding='utf-8', engine='python')
    print(train.shape)
    train['content'] = train['content'].fillna(u'文本 没有 内容')
    train['title'] = train['title'].fillna(u'标题 没有 内容')
    train = train.dropna()
    train = train.reset_index(drop=True)
    print(train.shape)
    train['title'] = train['title'].apply(lambda x: x.replace('|||', ' '))
    train['content'] = train['content'].apply(lambda x: x.replace('|||', ' '))
    train['title'] = train['title'].apply(lambda x: x.encode('utf-8'))
    train['content'] = train['content'].apply(lambda x: x.encode('utf-8'))
    train['word'] = train['word'].apply(lambda x: x.encode('utf-8'))
    print(train.shape)

    embeddings_index = pickle.open(path + 'embeddings_index.txt', 'rb')

    # 依次将文本、标题、候选词这些文本转换为序列表示
    print('token sequence...')
    tokenizer1 = Tokenizer(num_words=MAX_NB_WORDS)
    corpus = train[['content']]
    corpus = corpus.drop_duplicates()
    tokenizer1.fit_on_texts(corpus['content'])
    print('token done!')
    pickle.dump(tokenizer1, open(path + '/cache/tokenizer1.txt', 'wb'))
    # tokenizer1 = pickle.load(open(path + '/cache/tokenizer1.txt', 'rb'))

    print('transform text to sequence...')
    train_q1_word_seq = tokenizer1.texts_to_sequences(train['content'])

    word_index1 = tokenizer1.word_index
    nb_words = min(MAX_NB_WORDS, len(word_index1))

    # 初始化embedding矩阵
    word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index1.items():
        # print(word,i)
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector

    print('token word...')
    # if not os.path.exists(path + '/cache/tokenizer2.txt'):
    tokenizer2 = Tokenizer(num_words=MAX_NB_KEY_WORDS)
    tokenizer2.fit_on_texts(train['word'])
    pickle.dump(tokenizer2, open(path + '/cache/tokenizer2.txt', 'wb'))
    # tokenizer2 = pickle.load(open(path + '/cache/tokenizer2.txt', 'rb'))

    train_q2_word_seq = tokenizer2.texts_to_sequences(train['word'])
    # test_q2_word_seq = tokenizer2.texts_to_sequences(test['key_words'])

    word_index2 = tokenizer2.word_index
    nb_keywords = min(MAX_NB_KEY_WORDS, len(word_index2))
    keyword_embedding_matrix = np.zeros((nb_keywords + 1, EMBEDDING_DIM))
    for word, i in word_index2.items():
        # print(word,i)
        if i > MAX_NB_KEY_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            keyword_embedding_matrix[i] = embedding_vector

    MAX_NB_TITLE_WORDS = 200000
    # if not os.path.exists(path + '/cache/tokenizer3.txt'):
    tokenizer3 = Tokenizer(num_words=MAX_NB_TITLE_WORDS)
    tokenizer3.fit_on_texts(train['title'])
    pickle.dump(tokenizer3, open(path + '/cache/tokenizer3.txt', 'wb'))

    train_q3_word_seq = tokenizer3.texts_to_sequences(train['title'])

    word_index3 = tokenizer3.word_index
    nb_title_words = min(MAX_NB_TITLE_WORDS, len(word_index3))
    title_embedding_matrix = np.zeros((nb_title_words + 1, EMBEDDING_DIM))
    for word, i in word_index3.items():
        # print(word,i)
        if i > MAX_NB_TITLE_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            title_embedding_matrix[i] = embedding_vector

    MAX_WORD_SEQUENCE_LENGTH = 800
    MAX_TITLE_SEQUENCE_LENGTH = 50
    MAX_KEYWORD_SEQUENCE_LENGTH = 1

    train_q1_word_seq = pad_sequences(train_q1_word_seq, maxlen=MAX_WORD_SEQUENCE_LENGTH, value=0)
    train_q2_word_seq = pad_sequences(train_q2_word_seq, maxlen=MAX_KEYWORD_SEQUENCE_LENGTH, value=0)
    train_q3_word_seq = pad_sequences(train_q3_word_seq, maxlen=MAX_TITLE_SEQUENCE_LENGTH, value=0)

    feature = np.load(path + '/cache/train_features.npy')
    train_q4_word_seq = feature
    y = train.label.values

    # 分割训练集和验证集
    index = train.index
    train_idx, val_idx = train_test_split(index, test_size=0.1, shuffle=False)

    val_q1_word_seq, val_q2_word_seq, val_q3_word_seq, val_q4_word_seq = train_q1_word_seq[val_idx], train_q2_word_seq[
        val_idx], train_q3_word_seq[val_idx], train_q4_word_seq[val_idx]

    train_q1_word_seq, train_q2_word_seq, train_q3_word_seq, train_q4_word_seq = train_q1_word_seq[train_idx], \
                                                                                 train_q2_word_seq[train_idx], \
                                                                                 train_q3_word_seq[train_idx], \
                                                                                 train_q4_word_seq[train_idx]

    train_y, val_y = y[train_idx], y[val_idx]

    print('save data...')
    np.save(path + '/cache/train_q1_word_seq', train_q1_word_seq)
    np.save(path + '/cache/train_q2_word_seq', train_q2_word_seq)
    np.save(path + '/cache/train_q3_word_seq', train_q3_word_seq)
    np.save(path + '/cache/train_q4_word_seq', train_q4_word_seq)

    # np.save('../cache/train_q5_word_seq_', train_q5_word_seq_)

    np.save(path + '/cache/val_q1_word_seq', val_q1_word_seq)
    np.save(path + '/cache/val_q2_word_seq', val_q2_word_seq)
    np.save(path + '/cache/val_q3_word_seq', val_q3_word_seq)
    np.save(path + '/cache/val_q4_word_seq', val_q4_word_seq)

    np.save(path + '/cache/train_y', train_y)
    np.save(path + '/cache/val_y', val_y)

    print(word_embedding_matrix, keyword_embedding_matrix, title_embedding_matrix)
    np.save(path + '/cache/word_embedding_matrix', word_embedding_matrix)
    np.save(path + '/cache/keyword_embedding_matrix', keyword_embedding_matrix)
    np.save(path + '/cache/title_embedding_matrix', title_embedding_matrix)
