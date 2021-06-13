# coding:utf-8
import torch
from torch import nn
import numpy as np
from train_eval import train, test
from importlib import import_module
import argparse
import pandas as pd
import pickle
from data_loader import TextDataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
parser.add_argument('--mode', type=str, required=True, help='choose mode: train or test')
args = parser.parse_args()


def process_df(df):
    for f in ['title', 'content', 'word']:
        df[f] = df[f].fillna('')
        df[f] = df[f].apply(lambda x: x.lower())
    return df


if __name__ == '__main__':
    dataset = '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/zixun_tag/tag_extraction'  # 数据集
    idf_dic = pickle.load(open(dataset + '/data/idf_dic.txt', 'rb'))

    mode = args.mode  # train or test mode
    model_name = args.model  # bert

    x = import_module('models.' + model_name)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    config = x.Config(dataset, idf_dic)
    model = x.Model(config).to(config.device)
    model = nn.DataParallel(model)

    if mode != 'test':
        train_data = pd.read_csv(config.train_path, encoding='utf-8')
        train_data = process_df(train_data)
        dev_data = pd.read_csv(config.dev_path, encoding='utf-8')
        dev_data = process_df(dev_data)
        train_data = train_data.reset_index(drop=True)
        dev_data = dev_data.reset_index(drop=True)
        print(train_data.shape, dev_data.shape)

        train_dataset = TextDataset(config, train_data)
        dev_dataset = TextDataset(config, dev_data)

        train_sampler = RandomSampler(train_dataset)
        train_iter = DataLoader(train_dataset, sampler=train_sampler,
                                batch_size=config.train_batch_size, num_workers=4)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_iter = DataLoader(dev_dataset, sampler=dev_sampler,
                              batch_size=config.eval_batch_size, num_workers=4)
        train(config, model, train_iter, dev_iter)

    else:
        test_data = pd.read_csv(config.test_path, encoding='utf-8')
        test_data = process_df(test_data)
        print(test_data.shape)

        test_dataset = TextDataset(config, test_data)
        test_sampler = SequentialSampler(test_dataset)
        test_iter = DataLoader(test_dataset, sampler=test_sampler,
                               batch_size=config.eval_batch_size, num_workers=4)
        test(config, model, test_iter)

