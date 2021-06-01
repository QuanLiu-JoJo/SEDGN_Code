#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on Dec, 2020

@author: Chunkai Zhang
"""

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict') #参数出现时设为True
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)


def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.pkl', 'rb'))
    num_node = train_data[3]
    print(num_node)
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion) #分割数据集，返回俩元组
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.pkl', 'rb'))
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, shuffle=True) #Data数据类型
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset == 'diginetica': #item种类数
        n_node = num_node
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = num_node
    else:
        n_node = num_node

    model = trans_to_cuda(SessionGraph(opt, n_node)) #输入参数和节点总数，

    start = time.time()
    best_result_1 = [0, 0]
    best_epoch_1 = [0, 0]
    best_result_5 = [0, 0]
    best_epoch_5 = [0, 0]
    best_result_10 = [0, 0]
    best_epoch_10 = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit_1, mrr_1, hit_5, mrr_5, hit_10, mrr_10 = train_test(model, train_data, test_data) #输入：序列 + 标签
        flag_1 = 0
        if hit_1 >= best_result_1[0]:
            best_result_1[0] = hit_1
            best_epoch_1[0] = epoch
            flag_1 = 1
        if mrr_1 >= best_result_1[1]:
            best_result_1[1] = mrr_1
            best_epoch_1[1] = epoch
            flag_1 = 1
        print('Best Result:')
        flag_5 = 0
        if hit_5 >= best_result_5[0]:
            best_result_5[0] = hit_5
            best_epoch_5[0] = epoch
            flag_5 = 1
        if mrr_5 >= best_result_5[1]:
            best_result_5[1] = mrr_5
            best_epoch_5[1] = epoch
            flag_5 = 1
        print('\tRecall@5:\t%.4f\tMMR@5:\t%.4f\tEpoch:\t%d,\t%d'% (best_result_5[0], best_result_5[1], best_epoch_5[0], best_epoch_5[1]))
        flag_10 = 0
        if hit_10 >= best_result_10[0]:
            best_result_10[0] = hit_10
            best_epoch_10[0] = epoch
            flag_10 = 1
        if mrr_10 >= best_result_10[1]:
            best_result_10[1] = mrr_10
            best_epoch_10[1] = epoch
            flag_10 = 1
        print('\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tEpoch:\t%d,\t%d'% (best_result_10[0], best_result_10[1], best_epoch_10[0], best_epoch_10[1]))
        print('\tRecall@15:\t%.4f\tMMR@15:\t%.4f\tEpoch:\t%d,\t%d'% (best_result_1[0], best_result_1[1], best_epoch_1[0], best_epoch_1[1]))
        bad_counter += 1 - flag_10
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
