import numpy as np
import pandas as pd
import math
import csv

train_data = pd.read_csv('lab1_data/regression_dataset/train_set.csv')
validation_data = pd.read_csv('lab1_data/regression_dataset/validation_set.csv')
test_data = pd.read_csv('lab1_data/regression_dataset/test_set.csv')

def create_word_list(word_list, data):
	for sentence in data['Words (split by space)']:
		for word in sentence.split(' '):
			if word not in word_list:
				word_list.append(word)

def join_word_list(word_list1, word_list2):
	word_list = list(word_list1)
	for word in word_list2:
		if word not in word_list:
			word_list.append(word)
	return word_list

#处理得到one-hot矩阵
def create_one_hot(word_list, data):
	word_list_len = len(word_list)
	one_hot = list()
	#create one-hot
	for sentence in data['Words (split by space)']:
		tmp_one_hot = np.zeros(word_list_len)
		for word in sentence.split(' '):
			tmp_one_hot[word_list.index(word)] = 1
		one_hot.append(tmp_one_hot)
	return np.array(one_hot)

#计算两集合之间的距离
def calculate_distance(distance_list, validation_one_hot, train_one_hot):
	for i in range(len(validation_one_hot)):
		tmp_distance = list()
		for j in range(len(train_one_hot)):
			#distance calculating method
			#tmp_distance.append(math.sqrt(np.sum( (train_one_hot[j] - validation_one_hot[i])**2 ))) #欧式距离
			#tmp_distance.append(np.linalg.norm(train_one_hot[j] - validation_one_hot[i]))	#欧式距离
			tmp_distance.append( np.dot(train_one_hot[j], validation_one_hot[i])/( np.linalg.norm(train_one_hot[j])*np.linalg.norm(validation_one_hot[i]) ) ) #余弦距离
		distance_list.append(list(tmp_distance))

#处理获得训练集的可能性集合
def create_probability_list(data, probability_list):
    for i in range(len(data)):
        tmp_probability = list(data.iloc[i][1:])
        probability_list.append(tmp_probability)
    return np.array(probability_list)

#预测得到新数据的可能性集合
def get(index_list, validation_index, probability_list, distance, k):
		predict_probability = np.zeros(len(train_probability_list[0]))
		for i in range(k):
			predict_probability += ( probability_list[index_list[i]]/(distance_list[validation_index][index_list[i]]+0.0001) )
		#归一化
		tmp_sum = sum(predict_probability)
		predict_probability /= tmp_sum
		return predict_probability

#找到最优的K
def get_best_k(min_corr, best_k, k):
	predict_probability_list = list()
	for validation_index in range( len(distance_list) ):
		index_list = np.argsort(distance_list[validation_index])
		index_list = index_list[::-1]
		predict_probability_list.append(get(index_list, validation_index, train_probability_list, distance_list[validation_index], k))
	#计算相关系数
	cov_list = list()
	predict_probability_list = np.array(predict_probability_list)

	tmp = 0
	#对概率向量进行处理
	for i in range(len(predict_probability_list[0])):
		a = predict_probability_list[:,i]
		b = validation_probability_list[:,i]
		#计算相关系数
		tmp += np.corrcoef(a,b)[0][1]
	tmp = tmp / len(predict_probability_list[0])
	if tmp > min_corr:
		min_corr = tmp
		best_k = k
	return min_corr, best_k


#计算最优的k
#############################################
#KNN regression
emotion_list = list(train_data.iloc[0].index[1:])

#train_data process
train_word_list = list()
train_probability_list = list()
create_word_list(train_word_list, train_data)
train_probability_list = create_probability_list(train_data, train_probability_list)

#validation_data process
validation_word_list = list()
validation_probability_list = list()
create_word_list(validation_word_list, validation_data)
validation_probability_list = create_probability_list(validation_data, validation_probability_list)

#join word list
word_list = join_word_list(train_word_list, validation_word_list)
#create one-hot
train_one_hot = create_one_hot(word_list, train_data)					#train_one-hot
validation_one_hot = create_one_hot(word_list, validation_data) 	#validation_one-hot

distance_list = list()
calculate_distance(distance_list, validation_one_hot, train_one_hot)

#找k个最近的，并对可能性进行平均
#求最佳的k,从k=1开始，找到相关系数最大时对应的k
best_k = 1
max_corr = 0
for i in range(1, 30):	#30可变，个人认为在数据集长度的算术平方以内
	max_corr, best_k = get_best_k(max_corr, best_k, i)
print(max_corr, best_k)
'''

#############################################
#接下来的部分用于对test进行预测 可以将之前的注释掉，使用k=3直接开始测试

#test_data process
test_word_list = list()
test_probability_list = list()
create_word_list(test_word_list, test_data)

#join word list
word_list = join_word_list(train_word_list, test_word_list)
train_one_hot= create_one_hot(word_list, train_data)
test_one_hot = create_one_hot(word_list, test_data)

distance_list = list()
calculate_distance(distance_list, test_one_hot, train_one_hot)

#根据之前的计算知k=3时相关系数最高
#接下来的部分用于对test进行预测
best_k = 3
predict_probability_list = list()
for test_index in range( len(distance_list) ):
	index_list = np.argsort(distance_list[test_index])
	index_list = index_list[::-1]
	predict_probability_list.append(get(index_list, test_index, train_probability_list, distance_list[test_index], best_k))

result = pd.DataFrame(predict_probability_list)
result.to_csv('16337334_zhouqiheng_KNN_regression.csv', index=list(range(len(predict_probability_list))), header=emotion_list)
'''