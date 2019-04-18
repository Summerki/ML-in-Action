from math import log

def createDataSet():
    # 数据集特征
    # 年龄：0-青年，1-中年，2-老年
    # 有工作：0-否，1-有
    # 有自己的房子：0-否，1-有
    # 信贷情况：0-一般， 1-好， 2-非常好
    # 类别（是否给贷款）：no-否， yes-是
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]

    labels = ['不放贷', '放贷']  # 分类属性

    return dataSet, labels


# 根据给定数据集计算经验熵
def calcShannonEnt(dataSet):
    numEntires = len(dataSet)  # 数据集的行数
    labelCounts = {}  # 保存每个lable出现的次数

    for featVec in dataSet:
        currentLable = featVec[-1]
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable] = 0  # 相当于是一个初始化过程，后面再加1
        labelCounts[currentLable] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires  # 计算概率
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])  # 意思就是reducedFeatVec里面存放除了featVec[axis]那一行的数据
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最优特征
def chooseBestTeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 整个数据集的香农熵
    bestInfoGain = 0.0  # 信息增益
    bestFeature = -1  # 最优增益的索引值

    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]  # 把想要的特征值那一列的值都拿出来
        uniqueVals = set(featureList)
        newEntropy = 0.0  # 经验条件熵

        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy  # 信息增益
        print('第%d个特征的增益为%.3f'%(i, infoGain))

        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature





if __name__ == '__main__':
    dataSet, features = createDataSet()
    print('最优特征索引值：' + str(chooseBestTeatureToSplit(dataSet)))
    # print(dataSet)
    # print(calcShannonEnt(dataSet))