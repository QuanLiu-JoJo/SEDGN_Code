#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on Dec, 2020

@author: Chunkai Zhang
"""

import networkx as nx
import numpy as np


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail): #item_tail是填充序列尾部的数据
    us_lens = [len(upois) for upois in all_usr_pois] #所有序列的长度list
    len_max = max(us_lens) #序列的最大长度
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)] #填充尾部使之均为最长序列
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens] #掩码
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion): #划分交叉验证集，返回两个tuple
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32') # 输出 0 -> (n_sample - 1) 的array
    np.random.shuffle(sidx) #将sidx洗乱， 不输出
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0] #输入的序列
        inputs, mask, len_max = data_masks(inputs, [0]) #将序列填充至最大序列长度并返回序列，掩码，最大长度
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1]) #标签
        self.length = len(inputs) #有多少个输入序列
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size): #生成每个batch的索引array
        if self.shuffle: #洗乱序列的顺序
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size) #batch的数量
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch) #返回一个个batch大小的array组成的list
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))] #最后一项调整
        return slices

    def get_slice(self, i): #此处的i为slices，是个array数组
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i] #取出slice中的数据
        items, n_node, A, alias_inputs = [], [], [], [] #n_node记录每个序列各异的item数，items记录补到最长的各异点的list，A为所有session的出入度矩阵的list
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node) #所有序列异节点数极值
        for u_input in inputs:
            node = np.unique(u_input) #各异节点array， 后转为list
            items.append(node.tolist() + (max_n_node - len(node)) * [0]) #各异节点加补零的list
            u_A = np.zeros((max_n_node, max_n_node)) #长宽为最大各异节点数的矩阵
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0: #为零则序列结束， u_input并非是各异的
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1 #生成一个各异序列的有向图
            u_sum_in = np.sum(u_A, 0) #加成一行
            u_sum_in[np.where(u_sum_in == 0)] = 1 #为零的位置令为1
            u_A_in = np.divide(u_A, u_sum_in) #归一化，单列和为1
            u_sum_out = np.sum(u_A, 1) #加成一行
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out) #u_A的转置为出度矩阵
            u_A = np.concatenate([u_A_in, u_A_out]).transpose() #shape : max_n_node * (2 * max_n_node)
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input]) #u_input 在node中的位置 一个个list
        return alias_inputs, A, items, mask, targets
    # alias_inputs : u_input 在node中的位置 一个个array, batch_size * 序列长度
    # A为一个batch的出入度矩阵的list
    # items记录补到最长的各异点的list
    # mask : 补零后的某个slice的序列掩码
    #traget : 序列对应的结果