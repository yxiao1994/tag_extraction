# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer
import torch.nn.functional as F


class Config(object):
    """配置参数"""

    def __init__(self, dataset, idf_dic):
        # self.model_name = 'rubbish_bert'
        self.model_name = 'tag_extration_bert'                                # 测试集
        self.train_path = dataset + '/data/train_data.csv'  # 训练集
        self.dev_path = dataset + '/data/dev_data.csv'  # 验证集
        self.test_path = dataset + '/data/test_data.csv'  # 测试集
        self.pred_path = dataset + '/data/predict_data.csv'  # 预测结果
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        # self.save_path = dataset + '/model_backup/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.idf_dic = idf_dic
        self.require_improvement = 500  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 10  # epoch数
        self.train_batch_size = 128  # mini-batch大小
        self.eval_batch_size = 1024  # mini-batch大小
        self.learning_rate = 5e-3  # 学习率
        self.bert_path = '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/zixun_tag/bert_pretrain/'
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.input_feature_size = 13
        self.map_feature_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        # self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.feaure_map_layer = nn.Linear(config.input_feature_size, config.map_feature_size)
        self.fc = nn.Linear(config.hidden_size + config.map_feature_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq, seq_mask, seq_segment, dense_feature):
        _, pooled = self.bert(seq, attention_mask=seq_mask,
                              token_type_ids=seq_segment, output_all_encoded_layers=False)
        feature_ = F.relu(self.feaure_map_layer(dense_feature))
        merged = torch.cat([feature_, pooled], 1)
        out = self.sigmoid(self.fc(merged)).squeeze(1)
        return out

