# -*- coding: UTF-8 -*-
'''
@Project ：moleculeformer
@File    ：ratiofold.py
@Author  ：Mental-Flow
@Date    ：2024/4/12 11:06 
@introduction :
'''
import random
from sklearn.model_selection import train_test_split
from model.data import MoleDataSet
from collections import defaultdict

def print_label_distribution(set_name, dataset,log):
    label_count = defaultdict(int)
    for mol in dataset:
        label = mol.label[0]
        label_count[label] += 1
    log.debug(f"{set_name} label distribution: {dict(label_count)}")

def ratio_split(data,size,seed,log):
    # Shuffle the dataset to ensure randomness
    train_rate, val_rate, test_rate = size
    categorized_data = defaultdict(list)

    # 归类数据
    for mol in data.data:
        label = mol.label[0]
        categorized_data[label].append(mol)

    # 分别初始化训练集、验证集和测试集
    train_set, val_set, test_set = [], [], []

    # 分别对每个类别的数据进行分割
    for _, items in categorized_data.items():
        # 分配训练数据集和剩余数据集
        items_train, items_temp = train_test_split(items, train_size=train_rate, random_state=seed)

        # 计算测试集和验证集的大小
        test_val_size = 1.0 - train_rate
        val_relative = val_rate / test_val_size

        # 分配验证集和测试集
        items_val, items_test = train_test_split(items_temp, test_size=(1.0 - val_relative), random_state=seed)

        # 合并不同类别的数据集
        train_set.extend(items_train)
        val_set.extend(items_val)
        test_set.extend(items_test)

    # 如果需要记录日志
    if log:
        print_label_distribution("Training set", train_set,log)
        print_label_distribution("Validation set", val_set,log)
        print_label_distribution("Testing set", test_set,log)


    # 返回分割结果

    return MoleDataSet(train_set), MoleDataSet(val_set), MoleDataSet(test_set)

