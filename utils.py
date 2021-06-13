# coding:utf-8
#
# Copyright 2019 Tencent Inc.
# Author: Yang Xiao(mlhustxiao@tecent.com)
#
from datetime import timedelta
import time
import pandas as pd
import datetime
from sklearn.metrics import roc_auc_score


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def save_pred_file(config, test_prob):
    """保存预测结果"""
    test = pd.read_csv(config.test_path, encoding='utf-8')
    assert len(test) == len(list(test_prob))
    test['prob'] = test_prob
    print(roc_auc_score(test['label'], test['prob']))
    test['prob_rank'] = test.groupby('docid')['prob'].rank(ascending=False, method='first')
    res = test[(test.prob > 0.5) & (test.prob_rank <= 10)]
    res = res.sort_values(by=['docid', 'prob_rank'])
    data_dic = {}
    for i, row in res.iterrows():
        docid = row['docid']
        word = row['word']
        prob = round(row['prob'], 3)
        if docid not in data_dic:
            data_dic[docid] = word + ':' + str(prob)
        else:
            data_dic[docid] += (' ' + word + ':' + str(prob))
    res = pd.DataFrame({'docid': list(data_dic.keys()), 'tags': list(data_dic.values())})
    x = test[['docid', 'title', 'content']].drop_duplicates()
    res = pd.merge(res, x, on='docid', how='left')
    features = ['docid', 'title', 'content', 'tags']
    print('save result...')
    res[features].to_csv(config.pred_path, index=False, encoding='utf-8')

