import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore")

train_data = pd.read_csv('/Users/pp/pp_git/ai_lab/exp1/lab1_data/classification_dataset/train_set.csv')
validation_data = pd.read_csv('/Users/pp/pp_git/ai_lab/exp1/lab1_data/classification_dataset/validation_set.csv')

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

#train_data process
train_word_list = list()
train_sentence_list = list()
train_one_hot = list()

create_sentence_list(train_data, train_sentence_list)
create_word_list(train_word_list, train_sentence_list)
train_one_hot = create_one_hot(train_word_list, train_one_hot, train_sentence_list)

#validation_data process
validation_word_list = list()
validation_sentence_list = list()
validation_one_hot = list()

create_sentence_list(validation_data, validation_sentence_list)
validation_one_hot = create_one_hot(train_word_list, validation_one_hot, validation_sentence_list)


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
index_list = np.argsort(distance_list[0])
def get_K(distance_list, k):


