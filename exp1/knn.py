import numpy as np
import pandas as pd
import math

def create_sentence_list(data, sentence_list):
	for sentence in data['Words (split by space)']:
		sentence_list.append(sentence.split())

def create_word_list(word_list, sentence_list):
	for sentence in sentence_list:
		for word in sentence:
			if word not in word_list:
				word_list.append(word)

def create_one_hot(word_list, one_hot, sentence_list):
	word_list_len = len(word_list)

	#create one-hot
	for sentence in sentence_list:
		tmp_one_hot = list( 0 for i in range(word_list_len) )

		for word in sentence:
			if word in word_list:
				tmp_one_hot[word_list.index(word)] += 1
		
		one_hot.append( list(tmp_one_hot) )
	return np.array(one_hot)

def calculate_distance(distance_list, validation_one_hot, train_one_hot):
	for i in range(len(validation_one_hot)):
		tmp_distance = list()
		for j in range(len(train_one_hot)):
			#distance calculating method
			tmp_distance.append(math.sqrt(np.sum( (train_one_hot[j] - validation_one_hot[i])**2 )))
		distance_list.append(list(tmp_distance))
'''
def get(index_list, validation_index, accuracy_list, k):
	label_dict = {}
	for i in range(0, k):
		if train_data['label'][index_list[i]] not in label_dict:
			label_dict[train_data['label'][index_list[i]]] = 1
		else:
			label_dict[train_data['label'][index_list[i]]] += 1
	#predict via K value
	prediction = max(label_dict,key=label_dict.get)
	correct_answer = validation_data['label'][validation_index]
	if prediction == correct_answer:
		accuracy_list[k] += 1
        
train_data = pd.read_csv('/Users/pp/pp_git/ai_lab/exp1/lab1_data/classification_dataset/train_set.csv')
validation_data = pd.read_csv('/Users/pp/pp_git/ai_lab/exp1/lab1_data/classification_dataset/validation_set.csv')

#train_data process
train_word_list = list()
train_sentence_list = list()
train_one_hot = list()
create_sentence_list(train_data, train_sentence_list)
create_word_list(train_word_list, train_sentence_list)

#validation_data process
validation_word_list = list()
validation_sentence_list = list()
validation_one_hot = list()
create_sentence_list(validation_data, validation_sentence_list)
create_word_list(validation_word_list, validation_sentence_list)

#写错了，接下来要把word_list合并
#join word list
word_list = train_word_list
for word in validation_word_list:
	if word not in word_list:
		word_list.append(word)

#create one-hot
train_one_hot = create_one_hot(word_list, train_one_hot, train_sentence_list)			#train_one-hot
validation_one_hot = create_one_hot(word_list, validation_one_hot, validation_sentence_list) #validation_one-hot

#相似度
#save distance between validation_data and train_data
distance_list = list()
for i in range(len(validation_one_hot)):
	tmp_distance = list()
	for j in range(len(train_one_hot)):
		tmp_distance.append(math.sqrt(np.sum( (train_one_hot[j] - validation_one_hot[i])**2 )))
	distance_list.append(list(tmp_distance))

#determine the best value of K
#index_list
k = len(train_one_hot)
accuracy_list = list( 0 for i in range(k) )

for validation_index in range( len(distance_list) ):
	index_list = np.argsort(distance_list[validation_index])
	#get(index_list, validation_index, accuracy_list, 1)
	#0-k进行预测
	for i in range( 1, int(math.sqrt(k)) ):
		get(index_list, validation_index, accuracy_list, i)
total = len(validation_one_hot)
accuracy_list = [float(i)/total for i in accuracy_list]
'''


#KNN regression
def create_probability_list(data, probability_list):
    for i in range(len(data)):
        tmp_probability = list(data.iloc[i][1:])
        probability_list.append(tmp_probability)
    return np.array(probability_list)

train_data = pd.read_csv('/Users/pp/pp_git/ai_lab/exp1/lab1_data/regression_dataset/train_set.csv')
validation_data = pd.read_csv('/Users/pp/pp_git/ai_lab/exp1/lab1_data/regression_dataset/validation_set.csv')

#train_data process
train_word_list = list()
train_sentence_list = list()
train_one_hot = list()
train_probability_list = list()
create_sentence_list(train_data, train_sentence_list)
create_word_list(train_word_list, train_sentence_list)
train_probability_list = create_probability_list(train_data, train_probability_list)

#validation_data process
validation_word_list = list()
validation_sentence_list = list()
validation_one_hot = list()
validation_probability_list = list()
create_sentence_list(validation_data, validation_sentence_list)
create_word_list(validation_word_list, validation_sentence_list)
validation_probability_list = create_probability_list(validation_data, validation_probability_list)

#join word list
word_list = train_word_list
for word in validation_word_list:
	if word not in word_list:
		word_list.append(word)

train_one_hot= create_one_hot(word_list, train_one_hot, train_sentence_list)
validation_one_hot = create_one_hot(word_list, validation_one_hot, validation_sentence_list)

distance_list = list()
calculate_distance(distance_list, validation_one_hot, train_one_hot)

#找k个最近的，并对可能性进行平均
predict_probability_list = list()

def get(index_list, validation_index, probability_list, distance, k):
		predict_probability = np.zeros(len(train_probability_list[0]))
		for i in range(k):
			predict_probability += (probability_list[index_list[i]]/distance_list[validation_index][index_list[i]] + 1)
		#归一化
		tmp_sum = sum(predict_probability)
		predict_probability /= tmp_sum
		return predict_probability

def get_best_k(min_corr, best_k, k):
	predict_probability_list = list()
	for validation_index in range( len(distance_list) ):
		index_list = np.argsort(distance_list[validation_index])
		predict_probability_list.append(get(index_list, validation_index, train_probability_list, distance_list[validation_index], k))
	#计算相关系数
	cov_list = list()
	predict_probability_list = np.array(predict_probability_list)

	#对概率向量进行处理
	a = predict_probability_list[:,0]
	b = validation_probability_list[:,0]
	nan_a = np.isnan(a)
	a[nan_a] = 0
	nan_b = np.isnan(b)
	b[nan_b] = 0

	#计算相关系数
	tmp = np.cov(a,b)[0][1] /(np.var(a)*np.var(b))
	#print(tmp)
	if abs(tmp) < min_corr:
		min_corr = abs(tmp)
		best_k = k
		print(tmp)
	return min_corr, best_k

best_k = 1
min_corr = 20
for i in range(1, 20):
	min_corr, best_k = get_best_k(min_corr, best_k, i)

predict_probability_list = list()
for validation_index in range( len(distance_list) ):
	index_list = np.argsort(distance_list[validation_index])
	predict_probability_list.append(get(index_list, validation_index, train_probability_list, distance_list[validation_index], best_k))
print(predict_probability_list)


