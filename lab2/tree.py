import time
start = time.clock()
import math
import collections

#define node
class Node:
    def __init__(self, feature=None, value=None, decision=None):
        self.feature = feature
        #表示在一个feature类中选择哪一个特定的特征
        self.value = value
        #decision = 1预测为真
        self.decision = decision
        self.children_list = list()

    def print_child_list(self):
        if self.value != None:
            print('value:',self.value, 'choice:', self.decision)
        for child in self.children_list:
            child.print_child_list()

class Tree:
    def __init__(self, root):
        self.root = root

    def add_node(self, tmp_root, previous_feature):
        #new_child = Node(feature)
        #找到叶子
        if len(tmp_root.children_list) == 0:
            #在这里进行运算，找到最佳的特征
            min_h, index, choice_list = get_best(previous_feature)
            print(min_h)
            #储存一列中所有的value
            value_list = collections.Counter(list(train_data[index]))
            value_list = list(value_list.keys())

            for value_index in range(len(value_list)):
                tmp_root.children_list.append(Node(index, value_list[value_index], choice_list[value_index]))
            tmp_root.feature = index
        else:
            for index in range(len(tmp_root.children_list)):
                previous_feature.append( (tmp_root.children_list[index].feature, tmp_root.children_list[index].value) )
                self.add_node(tmp_root.children_list[index], previous_feature)
                #回溯
                previous_feature.pop()

    def print_tree(self):
        self.root.print_child_list()

def H(probability):
    if probability == 1 or probability == 0:
        return 0
    return (-1 * probability * math.log(probability, 2)) - (1 - probability) * math.log(1 - probability, 2)

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

#用于存储6种特征
feature_list = [1 for i in range(6)]

def join_previous_feature(previous_feature, value, feature_index):
    if len(previous_feature) >= 1:
        tmp = train_data[previous_feature[0][0]] == previous_feature[0][1]
        for i in previous_feature:
            tmp &= train_data[i[0]] == i[1]
        tmp &= train_data[feature_index] == value
        return tmp
    else: 
        return train_data[feature_index] == value

def get_h(feature_index, min_h, best_index, best_choice_list, previous_feature):
    feature_h = 0
    value_count = collections.Counter(list(train_data[feature_index]))
    choice_list = list()
    for value in value_count.keys():
        #计算特征为当前value时，结果为1的总数
        #index_list = train_data[(train_data[feature_index] == value)].index
        index_list = train_data[join_previous_feature(previous_feature, value, feature_index)].index
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
    #判断当前的特征是否更佳
    if feature_h < min_h:
        best_choice_list = list(choice_list)
        min_h = feature_h
        best_index = feature_index
    return min_h, best_index, best_choice_list

def get_best(previous_feature):
    best_choice_list = list()
    min_h = 1
    best_index = None
    for feature_index in range(len(feature_list)):
        if feature_index not in [previous_index[0] for previous_index in previous_feature]: #提取其中的元组的第一个元素
            min_h, best_index, best_choice_list = get_h(feature_index, min_h, best_index, best_choice_list, previous_feature)
    return min_h, best_index, best_choice_list

#index是选择的特征
root = Node()
my_tree = Tree(root)
my_tree.add_node(root, [])
my_tree.add_node(root, [])
my_tree.print_tree()



#当中是你的程序
elapsed = (time.clock() - start)
print("Time used:",elapsed)