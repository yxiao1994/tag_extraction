# coding: UTF-8
import torch
import torch.nn as nn
from utils import *
import numpy as np
import torch.nn.functional as F
from pytorch_pretrained.optimization import BertAdam


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_auc = float('-inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, batch in enumerate(train_iter):
            # print(trains)
            seq, seq_mask, seq_segment, dense_features, labels = \
                (x.to(config.device) for x in batch)
            labels = labels.squeeze()
            # print(dense_features, single_id_concat, multi_id_concat, mask_concat, labels)
            outputs = model(seq, seq_mask, seq_segment, dense_features)

            model.zero_grad()
            loss = F.binary_cross_entropy(outputs, labels.float())
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                dev_auc, dev_loss = evaluate(config, model, dev_iter)
                if dev_auc > dev_best_auc:
                    dev_best_auc = dev_auc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.3},  Val Loss: {2:>5.3},  Val AUC: {3:>6.3},Time: {4} {5}'
                print(msg.format(total_batch, loss.item(), dev_loss.item(), dev_auc, time_dif, improve))
                model.train()
            total_batch += 1
            # print(total_batch)
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    predict_all, auc, _ = evaluate(config, model, test_iter, test=True)
    print('test auc:' + str(auc))
    save_pred_file(config, predict_all)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in data_iter:
            seq, seq_mask, seq_segment, dense_features, labels = \
                (x.to(config.device) for x in batch)
            labels = labels.squeeze()
            outputs = model(seq, seq_mask, seq_segment, dense_features)
            loss = F.binary_cross_entropy(outputs, labels.float())
            loss_total += loss

            labels = labels.data.cpu().numpy()
            predict = outputs.data.cpu()

            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    auc = roc_auc_score(labels_all, predict_all)
    if test:
        return predict_all, auc, loss_total / len(data_iter)
    return auc, loss_total / len(data_iter)

