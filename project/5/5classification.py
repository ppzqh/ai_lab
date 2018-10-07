import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

#import labels
labels = []
with open('data/5/trainLabel.txt', 'r', encoding='utf-8') as trainLabel:
    for label in trainLabel:
        labels.append(int(re.sub(r'\n', '', label)))
labels = np.array(labels)

sentences = []
with open('data/5/trainData.txt', 'r', encoding='utf-8') as trainData:
    for sentence in trainData:
        sentences.append(re.sub(r'<br /><br />', ' ', sentence))

sentences = []
with open('data/5/trainData.txt', 'r', encoding='utf-8') as trainData:
    for sentence in trainData:
        sentences.append(re.sub(r'<br /><br />', ' ', sentence))

stemmer = SnowballStemmer("english")

for i in range(len(sentences)):
    sentences[i] = re.sub(r'["\)\(\*\.,!?``'']| \n', '', sentences[i]).split(' ')
    #变为小写,词形还原
    for j, word in enumerate(sentences[i]):
        sentences[i][j] = stemmer.stem(word.lower())
    #去重
    sentences[i] = set(sentences[i])


