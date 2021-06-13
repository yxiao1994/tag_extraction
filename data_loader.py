import logging
import torch
from torch.utils.data import Dataset
from get_features import *

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class TextDataset(Dataset):
    def __init__(self, config, df):
        # 数据预处理
        self.label = df['label'].values
        self.bert_tokenizer = config.bert_tokenizer
        self.title = df['title'].values
        self.content = df['content'].values
        self.word = df['word'].values
        self.max_seq1_len = 500
        self.max_seq2_len = 8
        self.max_seq_len = 512
        self.idf_dic = config.idf_dic

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        # 标签信息
        label = self.label[i]
        label = torch.LongTensor([label])

        title = self.title[i]
        content = self.content[i]
        word = self.word[i]
        dense_feature = get_features(word, title, content, self.idf_dic)

        content = content.replace(' ', '')
        title = title.replace(' ', '')

        sentence_1 = '#'.join([title, content])
        sentence_2 = word

        # 切词
        tokens_seq_1 = self.bert_tokenizer.tokenize(sentence_1)
        tokens_seq_1 = tokens_seq_1[:self.max_seq1_len]
        tokens_seq_2 = self.bert_tokenizer.tokenize(sentence_2)
        tokens_seq_2 = tokens_seq_2[:self.max_seq2_len]

        seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]'] + tokens_seq_2 + ['[SEP]']
        seq_segment = [0] * (len(tokens_seq_1) + 2) + [1] * (len(tokens_seq_2) + 1)
        # ID化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = seq_segment + padding
        # 对seq拼接填充序列
        seq += padding

        seq = torch.LongTensor(seq)
        seq_mask = torch.LongTensor(seq_mask)
        seq_segment = torch.LongTensor(seq_segment)
        dense_feature = torch.FloatTensor(dense_feature)

        return seq, seq_mask, seq_segment, dense_feature, label

