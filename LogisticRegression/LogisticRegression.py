# 逻辑回归算法实现

import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.font_manager import FontProperties

# testSet.txt文件说明
# 第一列-》X轴数值
# 第二列-》y轴数值
# 第三列-》分类标签;1是正样本；0是负样本


# 加载指定数据
def loadDataSet():
    dataMat = []  # 数据列表
    labelaMat = []  # 标签列表
    fr = open(r'./testSet.txt')
    for line in fr.readlines():
        # strip([chars])去除指定chars，默认去除首尾空格
        # split()默认以空格分隔
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelaMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelaMat



# 绘制初始数据集
def plotDataSet():
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]  # 多少行，也就是数据的个数
    # 正样本
    xcord1 = []
    ycord1 = []
    # 负样本
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



# sigmod函数
def sigmod(inX):
    return 1.0 / (1 + np.exp(-inX))



# 梯度上升法
'''
Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
    weights.getA() - 求得的权重数组(最优参数)
'''
def gradAscent(dataMatin, classLabels):
    # np.mat:https://www.jianshu.com/p/3a9c3a397932
    dataMatrix = np.mat(dataMatin)
    # transpose():转置
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)  # m行 n列
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmod(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    # getA():https://blog.csdn.net/qq_27396861/article/details/88538082
    return weights.getA()



# 绘制数据集，处理之后的
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    # 正样本
    xcord1 = []
    ycord1 = []
    # 负样本
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)
    x = np.arange(-3.0, 3.0, 0.1)
    # 因为决策边界为：w0+w1*x1+w2*x2=0
    # 所以x2 = (-w0-w1*x1) / w2
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# 改进1：将原来使用的梯度上升法更改为随机梯度上升法
# 随机梯度上升法：每次更新回归系数时不用所有样本，有效减小计算量
def stoGradAscent1(dataMatrix, classLabels, numIter = 150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # alpha在每次迭代的时候都会调整，并且，虽然alpha会随着迭代次数不断减小，但永远不会减小到0，因为这里还存在一个常数项
            alpha = 4 / (1.0 + j + i) + 0.01  # 降低alpha的大小，每次减小1/(j+i)
            # random.uniform:http://www.runoob.com/python/func-number-uniform.html
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmod(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])  # 删除已使用的样本
    return weights









if __name__ == '__main__':
    # plotDataSet()
    dataMat, labelMat = loadDataSet()
    # print(gradAscent(dataMat, labelMat))
    # weights = gradAscent(dataMat, labelMat)

    weights = stoGradAscent1(np.array(dataMat), labelMat)
    plotBestFit(weights)
