import re
import numpy as np

#读文件内容
semeval = open(r'semeval.txt', 'r')
content = semeval.read()
semeval.close()
#正则表达式获取所有句子
res = re.findall(r':\d*?\t(.*?)\n', content)
#获得文本数据集的全部单词
all_word = []
for sentence in res:
    for word in sentence.split(' '):
        if word not in all_word:
            all_word.append(word)
#计算TF矩阵和IDF向量
tf = np.zeros((len(res), len(all_word))) #(1246,2749)
for i, sentence in enumerate(res):
    for word in sentence.split(' '):
        index = all_word.index(word)
        tf[i][index] += 1
for each in tf:
    row_sum = each.sum()
    each /= row_sum
occur_doc_sum = (tf != 0).sum(axis=0)
idf = np.log(len(res) / (1 + occur_doc_sum))
#计算TF-IDF矩阵
tf_idf = tf * idf
#写文件
sub = open(r'16337329_ZYC_TFIDF.txt', 'w')
for each in tf_idf:
    s = ""
    for num in each:
        if num: s += str(num) + ' '
    s += '\n'
    sub.write(s)
sub.close()
