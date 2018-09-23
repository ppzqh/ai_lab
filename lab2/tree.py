import time
start = time.clock()
import math
import collections

#define node
class Node:
    def __init__(self, feature):
        self.feature = feature
        #choice = 1预测为真
        self.choice = 1
        self.children_list = list()

    def print_child_list(self):
        print(self.feature)
        for child in self.children_list:
            child.print_child_list()

class Tree:
    def __init__(self, root):
        self.root = root

    def add_node(self, tmp_root, feature_list):
        #new_child = Node(feature)
        #找到叶子
        if len(tmp_root.children_list) == 0:
            for feature in feature_list:
                tmp_root.children_list.append(Node(feature))
        else:
            for index in range(len(tmp_root.children_list)):
                self.add_node(tmp_root.children_list[index], feature_list)


    def print_tree(self):
        self.root.print_child_list()

def H(probability):
    if probability == 1 or probability == 0:
        return 0
    return (-1 * probability * math.log(probability, 2)) - (1 - probability) * math.log(1 - probability, 2)


'''
my_tree.add_node(root, 'income')
my_tree.add_node(root, 'student')
my_tree.add_node(root.children_list[0], '1')
my_tree.print_tree()
'''

#read file
import numpy as np
import pandas as pd
train_data = pd.read_csv('/Users/pp/pp_git/ai_lab/lab2/lab2_data/Car_train.csv', header=None) #header=None表示没有列索引
#加上列索引
train_data = train_data.reindex(columns = [0, 1, 2, 3, 4, 5, 6])

#总长度
data_length = len(train_data)

#计算result的熵
false_len = len(train_data[train_data[6] == 0])
H_result = H(float(false_len / len(train_data)))


#计算特征0(之后改为i)为low时，结果为1的总数 
#index_list = train_data[(train_data[0] == 'low')].index
#train_data[6][index_list].sum()

#找出特征0(之后改为i)中不重复的元素
a = train_data[0].drop_duplicates()

#用于标记feature是否已经被使用
feature_list = [1 for i in range(6)]

def get_h(feature_index, min_h, best_index, best_choice_list):
    feature_h = 0
    value_count = collections.Counter(list(train_data[feature_index]))
    choice_list = list()
    for value in value_count.keys():
        #计算特征为当前value时，结果为1的总数
        index_list = train_data[(train_data[feature_index] == value)].index
            #计算1的个数
        tmp_true_count = train_data[6][index_list].sum()
            #取0，1中出现次数多的
        choice_count = max(value_count[value] - tmp_true_count, tmp_true_count)
        #1为yes 0为no
        choice_list.append(choice_count == value_count[value])
        '''
        这里使用的是信息增益
        '''
        tmp_probability = float( choice_count / value_count[value])
        feature_h += float(value_count[value] / data_length) * H(tmp_probability)

    if feature_h < min_h:
        best_choice_list = choice_list
        min_h = feature_h
        best_index = feature_index
    return min_h, best_index

def get_best():
    best_choice_list = list()
    min_h = 1
    best_index = 0
    for feature_index in range(len(feature_list)):
        if feature_list[feature_index] == 1:
            min_h, best_index = get_h(feature_index, min_h, best_index, best_choice_list)
    return min_h, best_index, best_choice_list

min_h, index, choice_list = get_best()
feature_list = list(collections.Counter(list(train_data[index])).keys())
print(feature_list)
root = Node(index)
my_tree = Tree(root)
my_tree.add_node(root, feature_list)
my_tree.print_tree()


#当中是你的程序
elapsed = (time.clock() - start)
print("Time used:",elapsed)