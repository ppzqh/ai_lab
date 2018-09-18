import numpy as np
import pandas as pd
import math
#read data from files
train_data = pd.read_csv('lab1_data/classification_dataset/train_set.csv')
validation_data = pd.read_csv('lab1_data/classification_dataset/validation_set.csv')
test_data = pd.read_csv('lab1_data/classification_dataset/test_set.csv')

def create_word_list(word_list, data):
	for sentence in data['Words (split by space)']:
		for word in sentence.split(' '):
			if word not in word_list:
				word_list.append(word)

def create_one_hot(word_list, data):
	word_list_len = len(word_list)
	one_hot = list()
	#create one-hot
	for sentence in data['Words (split by space)']:
		tmp_one_hot = np.zeros(word_list_len)
		for word in sentence.split(' '):
			if word in word_list:
				tmp_one_hot[word_list.index(word)] += 1
		one_hot.append(tmp_one_hot)
	return np.array(one_hot)

def join_word_list(word_list1, word_list2):
	word_list = list(word_list1)
	for word in word_list2:
		if word not in word_list:
			word_list.append(word)
	return word_list
'''
calculate这里可改进
'''
def calculate_distance(distance_list, validation_one_hot, train_one_hot):
	for i in range(len(validation_one_hot)):
		tmp_distance = list()
		for j in range(len(train_one_hot)):
			#distance calculating method
			tmp_distance.append(math.sqrt(np.sum( (train_one_hot[j] - validation_one_hot[i])**2 )))
			#tmp_distance.append( np.dot(train_one_hot[j], validation_one_hot[i])/( np.linalg.norm(train_one_hot[j])*np.linalg.norm(validation_one_hot[i]) ) ) #余弦距离
		distance_list.append(tmp_distance)

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

def predict(index_list, validation_index, k):
	label_dict = {}
	for i in range(0, k):
		if train_data['label'][index_list[i]] not in label_dict:
			label_dict[train_data['label'][index_list[i]]] = 1
		else:
			label_dict[train_data['label'][index_list[i]]] += 1
	#predict via K value
	prediction = max(label_dict,key=label_dict.get)
	return prediction

#计算最优的k
#############################################
#train_data process
train_word_list = list()
create_word_list(train_word_list, train_data)
'''
#validation_data process
validation_word_list = list()
create_word_list(validation_word_list, validation_data)

#接下来要把word_list合并
#join word list
word_list = join_word_list(train_word_list, validation_word_list)

#create one-hot
train_one_hot = create_one_hot(word_list, train_data)					#train_one-hot
validation_one_hot = create_one_hot(word_list, validation_data) 	#validation_one-hot

#相似度
#save distance between validation_data and train_data
distance_list = list()
calculate_distance(distance_list, validation_one_hot, train_one_hot)

#determine the best value of K
#index_list
k = len(train_one_hot) #k是最大
accuracy_list = list( 0 for i in range(int(math.sqrt(k))) )

for validation_index in range( len(distance_list) ):
	index_list = np.argsort(distance_list[validation_index])
	#index_list = index_list[::-1]
	#0-sqrt(k)进行预测
	for i in range( 1, int(math.sqrt(k)) ):
		get(index_list, validation_index, accuracy_list, i)

total = len(distance_list)
accuracy_list = [float(i)/total for i in accuracy_list]
best_k = accuracy_list.index(max(accuracy_list))
print(best_k)
print(accuracy_list)
'''
#由计算得到best_k = 12
#make prediction for test 可将前面的注释掉
#############################################
#test_data process
test_word_list = list()
create_word_list(test_word_list, test_data)

#接下来要把word_list合并
#join word list
word_list = join_word_list(train_word_list, test_word_list)

#create one-hot
train_one_hot = create_one_hot(word_list, train_data)					#train_one-hot
test_one_hot = create_one_hot(word_list, test_data) 	#validation_one-hot
distance_list = list()
calculate_distance(distance_list, test_one_hot, train_one_hot)

best_k = 12
prediction = list()
for test_index in range( len(distance_list) ):
	index_list = np.argsort(distance_list[test_index])
	prediction.append(predict(index_list, test_index, best_k))

result = pd.DataFrame({'Words (split by space)':test_data['Words (split by space)'], 'label': prediction})
result.to_csv('16337334_zhouqiheng_KNN_classification.csv', index=list(range(len(prediction))))