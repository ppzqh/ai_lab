import numpy as np

f = open('lab1_data/semeval.txt')
#word_dict = dict()
sentence_list = list()
word_list = list()

#file operation
while True:
	line = f.readline()
	if not line:
		break
	line = line.split('\t', 2)
	power_str = line[1]
	line[2] = line[2].rstrip('\n')
	line_word = line[2].split(' ')
	sentence_list.append(line_word)
	for i in line_word:
		if i not in word_list:
			word_list.append(i)

#create IDF & TF
one_hot = list()
word_list_len = len(word_list)
word_count = list( 0 for i in range(word_list_len) )
sentence_list_len = len(sentence_list)

for sentence in sentence_list:
	#print(len(sentence))
	tmp_one_hot = list( 0 for i in range(word_list_len) )

	for word in sentence:
		tmp_one_hot[word_list.index(word)] += 1
	
	#word count---IDF
	for index in range(word_list_len):
		if tmp_one_hot[index] != 0:
			word_count[index] += 1

	#TF		
	tmp_one_hot = [float(i) / float(len(sentence)) for i in tmp_one_hot]
	one_hot.append( list(tmp_one_hot) )

TF = np.array(one_hot)
IDF = np.array(word_count)
#get IDF
IDF = np.log2( sentence_list_len/(IDF+1) )
TF_IDF = TF * IDF

result = list()

for row_index in range(len(TF_IDF)):
	tmp = list()
	for i in range(len(sentence_list[row_index])):
		#按顺序找到序号
		appear_index = word_list.index(sentence_list[row_index][i])
		tmp.append(TF_IDF[row_index][appear_index])
	result.append(tmp)

output = open('16337334_zhouqiheng_TFIDF.txt', 'w')
for i in result:
	tmp_str = str(i);
	tmp_str = tmp_str.rstrip(']')
	tmp_str = tmp_str.lstrip('[')
	output.write(tmp_str)
	output.write('\n')
output.close()

