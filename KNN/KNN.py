import numpy as np
import operator

def createDataSet():
    # 四组二维数组
    group = np.array([
        [1,101],
        [5,89],
        [108,5],
        [115,8]
    ])

    # 四组特征的标签
    labels = ['爱情片', '爱情片', '动作片', '动作片']

    return group, labels

'''
KNN算法

inX：用于分类的数据（测试集）
dataSet：用于训练的数据（训练集）
labels：分类标签
k:KNN参数
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # tile函数：https://blog.csdn.net/ooxxshaso/article/details/81383315
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # argsort()->https://www.cnblogs.com/yyxf1413/p/6253995.html
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        # 字典的get函数;http://www.runoob.com/python/att-dictionary-get.html    后面的0代表值不存在就返回0
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    #创建数据集
    group, labels = createDataSet()
    #测试集
    test = [101,20]
    #kNN分类
    test_class = classify0(test, group, labels, 3)
    #打印分类结果
    print(test_class)