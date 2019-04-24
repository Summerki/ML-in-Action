# 朴素贝叶斯的实现
# 背景：在线留言社区实现侮辱性语言标记
# 侮辱类-1     非侮辱类-0

import numpy as np
from functools import reduce
# functools.reduce用法;http://www.cnblogs.com/alan-babyblog/p/5194399.html


# 创建实验样本
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec


# 根据vocabList词汇表将inputSet向量化，向量的每个元素为1或0
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList) # 创建一个所含元素都是0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print('the word : %s is not in  my Vocabulary!'%word)
    return returnVec



# 将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)



# 朴素贝叶斯分类器训练函数
'''
Parameters:
    trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
    p0Vect - 侮辱类的条件概率数组
    p1Vect - 非侮辱类的条件概率数组
    pAbusive - 文档属于侮辱类的概率
'''
def trainNB0(trainMatrix, trainCategory):
    #region
    # numTrainDocs = len(trainMatrix)  # 训练的文档数目
    # numWords = len(trainMatrix[0])  # 每片文档的词条数
    # pAbusive = sum(trainCategory) / float(numTrainDocs)  # 侮辱性文档个数 / 文档数目   文档属于侮辱类的概率
    # p0Num = np.zeros(numWords)
    # p1Num = np.zeros(numWords)
    # p0Denom = 0.0
    # p1Denom = 0.0
    # for i in range(numTrainDocs):
    #     if trainCategory[i] == 1:
    #         p1Num += trainMatrix[i]
    #         p1Denom += sum(trainMatrix[i])
    #     else:
    #         p0Num += trainMatrix[i]
    #         p0Denom += sum(trainMatrix[i])
    #
    # p1Vect = p1Num / p1Denom  # 在侮辱性词汇中某个侮辱性词汇的条件概率   p(xxx|侮辱类)
    # p0Vect = p0Num / p0Denom  # 在非侮辱性词汇中某个非侮辱性词汇的概率   p(xxx|非侮辱类)
    #
    # return p0Vect, p1Vect, pAbusive
    #endregion
    numTrainDocs = len(trainMatrix)  # 计算训练的文档数目
    numWords = len(trainMatrix[0])  # 计算每篇文档的词条数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于侮辱类的概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)  # 创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0  # 分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)  # 取对数，防止下溢出
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive  # 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率



# 朴素贝叶斯分类器分类函数
'''
Parameters:
	vec2Classify - 待分类的词条数组
	p0Vec - 侮辱类的条件概率数组
	p1Vec -非侮辱类的条件概率数组
	pClass1 - 文档属于侮辱类的概率
Returns:
	0 - 属于非侮辱类
	1 - 属于侮辱类
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #region
    # # 注：下面式子的理解：
    # # p1 = p('侮辱类'|['xxx', 'xxx',...]) = p['侮辱类'] * (p(['xxx', 'xxx', ...] | '侮辱类') / p['xxx', 'xxx', ...])
    # # 根据博客讲的，由于（）内的分母都是p['xxx', 'xxx', ...]所以不用计算这个也可以进行比较大小
    # # 但是一旦其中有一个概率p('xxx' | '侮辱类')为0的话，那么整项都为0了也就会出错
    # # 所以需要对trainNB0函数进行改变
    # p1 = reduce(lambda x,y:x*y, vec2Classify * p1Vec) * pClass1
    # p0 = reduce(lambda x,y:x*y, vec2Classify * p0Vec) * (1.0 - pClass1)
    # print('p0:', p0)
    # print('p1:', p1)
    # if p1 > p0:
    #     return 1
    # else:
    #     return 0
    #endregion

    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # 对应元素相乘。log(A * B) = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0




# 测试朴素贝叶斯分类器
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')
    else:
        print(testEntry, '属于非侮辱类')
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')
    else:
        print(testEntry, '属于非侮辱类')


if __name__ == '__main__':
    # postingList, classVec = loadDataSet()
    # # for each in postingList:
    # #     print(each)
    # #
    # # print(classVec)
    #
    #
    # # postingList是原始的词条列表
    # print('postingList:\n', postingList)
    # myVocabList = createVocabList(postingList)
    # # myVocabList是所有单词出现的集合，没有重复的元素
    # print('myVocabList:\n', myVocabList)
    # trainMat = []
    # for postingDoc in postingList:
    #     trainMat.append(setOfWords2Vec(myVocabList, postingDoc))
    # # 词汇表是用来将词条向量化的，一个单词在词汇表中出现过一次，那么就在相应位置记作1，如果没有出现就在相应位置记作0。trainMat是所有的词条向量组成的列表。
    # print('trainMat:\n', trainMat)
    #
    # # p1V存放的就是各个单词属于侮辱类的条件概率。pAb就是先验概率
    # p0V, p1V, pAb = trainNB0(trainMat, classVec)
    # print('p0V:\n', p0V)
    # print('p1V:\n', p1V)
    # print('classVec:\n', classVec)
    # print('pAb:\n', pAb)



    testingNB()