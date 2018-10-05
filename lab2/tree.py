import time
start = time.clock()
import math
import collections

#define node
class Node:
    def __init__(self, feature=None, value=None, decision=None, value_list=None):
        self.feature = feature
        #表示在一个feature类中选择哪一个特定的特征
        self.value = value
        #decision = 1预测为真
        self.decision = decision
        self.children_list = list()
        #储存该节点属性对应的取值集合
        self.value_list = value_list
        self.terminate = False

    def print_child_list(self):
        for child in self.children_list:
            child.print_child_list()
        if self.value != None:
            print('value:',self.value, 'choice:', self.decision)

class Tree:
    def __init__(self):
        self.root = Node()

    def add_node(self, all_finished, train_data, feature_num, tmp_root, previous_feature, previous, method):
        #若terminate为0，说明已经分完了，终止
        if tmp_root.terminate == True:
            return
        #找到叶子
        if len(tmp_root.children_list) == 0:
            #在这里进行运算，找到最佳的特征
            min_h, index, choice_list = get_best(train_data, feature_num, previous_feature, previous, method)
            #将terminate标志为true
            if min_h == 0:
                tmp_root.terminate = True
                all_finished |= previous_feature
                
            if tmp_root.feature == None:
                #储存一列中所有的value
                tmp_root.value_list = list(collections.Counter(list(train_data[index])).keys())
                tmp_root.feature = index

            for value_index in range(len(tmp_root.value_list)):
                tmp_root.children_list.append(Node(value=tmp_root.value_list[value_index], decision=choice_list[value_index]))
        
        else:
            if tmp_root.decision == -1:
                return

            for index in range(len(tmp_root.children_list)):
                previous.append(tmp_root.feature)
                self.add_node(all_finished, train_data, feature_num, tmp_root.children_list[index], previous_feature & set(train_data[train_data[tmp_root.feature] == tmp_root.children_list[index].value].index), previous, method)
                previous.pop()
                
    def train(self, train_data, method):
        feature_num = train_data.shape[1] - 1
        #加上列索引
        train_data = train_data.reindex(columns = range(feature_num + 1))
        all_finished = set()
        #两种终止条件，速度好像差不多
        while(len(all_finished) < len(train_data)):
            self.add_node(all_finished, train_data, feature_num, self.root, set(range(len(train_data))), [], method)

    def predict_one(self, tmp_root, to_predict, decision_index, result, choice):
        if len(tmp_root.children_list) == 0:
            #选择是训练还是预测
            if choice == 'train':
                result[0] += int(tmp_root.decision == to_predict[decision_index])
            elif choice == 'predict':
                result.append(tmp_root.decision)
            return

        for child in tmp_root.children_list:
            if child.value == to_predict[tmp_root.feature]:
                if child.decision == -1:
                    if choice == 'train':
                        result[0] += int(tmp_root.decision == to_predict[decision_index])
                    elif choice == 'predict':
                        result.append(tmp_root.decision)
                    return

                self.predict_one(child, to_predict, decision_index, result, choice)

    def predict_all(self, test_data, result, choice):
        decision_index = test_data.shape[1] - 1
        test_data = test_data.reindex(columns = range(decision_index + 1))
        for tmp_data_index in range(len(test_data)):
            self.predict_one(self.root, test_data.loc[tmp_data_index], decision_index, result, choice)
    
    def fit(self, test_data, choice):
        if choice == 'train':
            result = [0]
        elif choice == 'predict':
            result = list()
        else:
            print('Unknown choice')
            return
        self.predict_all(test_data, result, choice)
        return result

    def print_tree(self):
        self.root.print_child_list()

def H(probability):
    if probability == 1 or probability == 0:
        return 0
    return (-1 * probability * math.log(probability, 2)) - (1 - probability) * math.log(1 - probability, 2)

def gini(probability):
    return probability * (1 - probability)

#在这里加入别的计算方法
def get_h(train_data, feature_index, min_h, best_index, best_choice_list, previous_feature, feature_num, method):
    feature_h = 0
    value_count = collections.Counter(list(train_data[feature_index]))
    choice_list = list()
    for value in value_count.keys():
        #计算特征为当前value时，结果为1的总数
        #index 使用set来获得 每次将新的集合与旧的求交集
        index_list = previous_feature & set(train_data[train_data[feature_index] == value].index)
        #如果没有该类别的元素，将选择置为-1，以便之后处理
        if len(index_list) == 0:
            choice_list.append(-1)
            continue
            #计算1的个数
        tmp_true_count = train_data[feature_num][index_list].sum()
            #取0，1中出现次数多的
        choice_count = max(len(index_list) - tmp_true_count, tmp_true_count)
        #True为1 False为0
        choice_list.append(choice_count == tmp_true_count)
        #之前的计算公式有错，进行了修改
        tmp_probability = float( choice_count / len(index_list) )
        if method == 'CART':
            feature_h += float(len(index_list) / len(previous_feature)) * gini(tmp_probability)   
        else:
            feature_h += float(len(index_list) / len(previous_feature)) * H(tmp_probability)
    '''
    信息增益率
    '''
    if method == 'C4.5':
        splitinfo = 0
        for i in value_count.keys():
            tmp_probability = float(value_count[i])/len(data)
            splitinfo += tmp_probability * math.log(tmp_probability, 2)
        feature_h /= -1 * splitinfo
    #判断当前的特征是否更佳
    if feature_h < min_h:
        best_choice_list = list(choice_list)
        min_h = feature_h
        best_index = feature_index
    return min_h, best_index, best_choice_list

def get_best(train_data, feature_num, previous_feature, previous, method):
    best_choice_list = list()
    min_h = 10
    best_index = None
    for feature_index in range(feature_num):
        if feature_index not in [previous_index for previous_index in previous]:
            min_h, best_index, best_choice_list = get_h(train_data, feature_index, min_h, best_index, best_choice_list, previous_feature, feature_num, method)
    return min_h, best_index, best_choice_list

def find_leaf(tmp_root):
    if len(tmp_root.children_list) > 0:
        for i in tmp_root.children_list:
            find_leaf(i)
    else:
        print(tmp_root.decision)

def k_fold_cross_validation(k, data, method):
    fold_length = int(len(data)/k)
    accuracy = 0
    for i in range(k):
        #处理训练集和验证集
        train_data = data.iloc[:i*fold_length].append(data.iloc[(i+1)*fold_length: ]).reset_index(drop=True)
        validation_data = data.iloc[i*fold_length: (i+1)*fold_length].reset_index(drop=True)
        my_tree = Tree()
        my_tree.train(train_data, method)
        accuracy += (my_tree.fit(validation_data, 'train')[0])/len(validation_data)
    print('k='+str(k)+',', 'accuracy='+str(float(accuracy)/k))

#read file
import numpy as np
import pandas as pd
data = pd.read_csv('/Users/pp/pp_git/ai_lab/lab2/lab2_data/Car_train.csv', header=None) #header=None表示没有列索引
methods = ['ID3', 'C4.5', 'CART']

#train for the best k
for method in methods:
    print('Model:', method)
    for k in range(2, 10):
        k_fold_cross_validation(k, data, method)
'''
test_data = pd.read_csv('/Users/pp/pp_git/ai_lab/lab2/lab2_data/Car_test.csv', header=None)
for method in methods:
    print('Model:',method)
    my_tree = Tree()
    my_tree.train(data, method)
    result = my_tree.fit(test_data, 'predict')
    print(result)
    break
'''
#当中是你的程序
elapsed = (time.clock() - start)
print("Time used:",elapsed)