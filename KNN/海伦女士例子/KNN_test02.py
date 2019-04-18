import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
import operator

'''
海伦数据：
3个特征：
每年获得的飞行常客里程数
玩视频游戏所消耗时间百分比
每周消费的冰淇淋公升数

还有最后一列是'target'结果

'''



# 准备数据，数据解析
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0

    for line in arrayOLines:
        line = line.strip()
        listFormLine = line.split('\t')
        returnMat[index, :] = listFormLine[0:3]

        if listFormLine[-1] == 'didntLike':
            classLabelVector.append(1)
        if listFormLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        if listFormLine[-1] == 'largeDoses':
            classLabelVector.append(3)

        index+=1

    return returnMat, classLabelVector


# 分析数据，数据可视化
def showdatas(datingDataMat, datingLabels):
    # 给matplotlib添加中文字体：https://www.jianshu.com/p/ff340060e140
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13,8))

    numberOfLabels = len(datingLabels)
    LabelColors = []
    for i in datingLabels:
        if i == 1:
            LabelColors.append('black')
        if i == 2:
            LabelColors.append('orange')
        if i == 3:
            LabelColors.append('red')


    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelColors, s=15, alpha=0.5)
    # py中字符串前面加u、r的意思https://blog.csdn.net/ff55fff/article/details/77504575
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占', FontProperties=font)
    # setp == set property
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.setp.html
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')


    # 设置图例
    # https://www.jianshu.com/p/4a6565891faf
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')

    # 添加图例
    # 关于legend：https://blog.csdn.net/qq_24694761/article/details/79176789
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    # 解决新版Pycharm中Matplotlib图像不在弹出独立的显示窗口https://blog.csdn.net/u010472607/article/details/82290159
    plt.show()



# 准备数据：数据归一化
def autoNorm(dataSet):
    # min(0) max(0):https://blog.csdn.net/mxhsyyd/article/details/80312045
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]

    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))

    return normDataSet, ranges, minVals



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




# 测试算法，验证分类器
def datingClassTest():
    filename = "./datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)

    # 取所有的数据10%作为测试集
    hoRatio = 0.10
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio) # 测试集的个数
    errorCount = 0.0  # 分类错误计数

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print("分类结果：%d\t真实类别：%d" %(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0

    print("错误率：%f%%"%(errorCount / float(numTestVecs) * 100))



# 使用算法，构建完整可用算法
def classifyPerson():
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    percentTats = float(input('玩视频游戏所耗费时间百分比：'))
    ffMiles = float(input('每年获得的飞行常客里程数：'))
    iceCream = float(input('每周消费的冰激凌公升数：'))

    filename = './datingTestSet.txt'
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 将输入的生成numpy数组，测试集
    inArr = np.array([percentTats, ffMiles, iceCream])
    norminArr = (inArr - minVals) / ranges  # 对测试集也做正则化
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    print('你可能%s这个人'%(resultList[classifierResult - 1]))

if __name__ == '__main__':
    # filename = './datingTestSet.txt'
    # datingDataMat, datingLabels = file2matrix(filename)
    # normDataSet, ranges, minVals = autoNorm(datingDataMat)
    # print(normDataSet)
    # print(ranges)
    # print(minVals)

    # datingClassTest()

    classifyPerson()

    # showdatas(datingDataMat, datingLabels)
    # print(datingDataMat)
    # print(datingLabels)