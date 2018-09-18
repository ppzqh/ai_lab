# <center>人工智能实验lab1<center>

<center>16337334 周启恒</center>

**<center>中山大学数据科学与计算机学院</center>**

**<center>计算机科学与技术 人工智能</center>**

**<center>本科生实验报告</center>**

**<center>课程名称：Artificial Intelligence</center>**

| 教学班级 | 16级计科二班 | 专业（方向） | 计算机科学与技术 |
| -------- | ------------ | ------------ | ---------------- |
| 学号     | 16337334     | 姓名         | 周启恒           |

## （1）算法原理

### TFIDF

1. 读取文件，将句子分成单独的词汇，并汇总成一个word_list。
2. 根据word_list对每个句子构建一个one-hot矩阵。
3. 根据公式，将one-hot矩阵处理为TF矩阵，对word_list进行处理得到IDF矩阵
4. TF*IDF得到TF-IDF



### KNN-classification & KNN-regression

**算法理解：**

首先通过对训练集和验证集的处理，得到one-hot矩阵。

取验证集中的一项，计算它与训练集所有数据之间的距离，选择L组最近的数据，并找出L组中类别的众数，将它作为验证集此项的预测结果。

1. classification：对验证集所有数据进行上述操作，得到一个预测结果集（离散），直接与真实结果进行比对，计算准确率，准确率可代表k=L时预测的效果。
2. regression：对验证集所有数据进行上述操作，得到一个预测可能性集（连续），计算它与真实结果集的相关系数，这个相关系数对应的是k=L时预测的效果。

遍历L，可找到一个最佳的k，将它作为最终对测试集分类时的参数。

##（2）流程图

![image-20180917224458191](/Users/pp/Library/Application Support/typora-user-images/image-20180917224458191.png)

## （3）关键代码截图

## （4）创新点&优化

## （5）实验结果展示



## （6）评测指标展示



