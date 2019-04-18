import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN

# sklearn小试牛刀：将32X32的二进制图像转换为1X1024向量
def img2vector(filename):
    # 创建1x1024零向量
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline() # 每次读一行
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])

    return returnVect


# 手写数字分类测试
def handwritingClassTest():
    hwLabels = []  # 测试集的label
    trainingFileList = listdir('./trainingDigits')
    m = len(trainingFileList)  # 返回指定文件夹下的文件个数
    trainingMat = np.zeros((m, 1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = img2vector('./trainingDigits/%s'%(fileNameStr))

    # 构建KNN分类器
    neigh = KNN(n_neighbors=3, algorithm='auto')
    neigh.fit(trainingMat, hwLabels)
    testFileList = listdir('./testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        classifierResult = neigh.predict(vectorUnderTest)
        print('分类返回结果为%d\t真实结果%d'%(classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1

    print("总共错了%d个数据\n错误率为%f%%"%(errorCount, errorCount / mTest * 100))



if __name__ == '__main__':
    handwritingClassTest()
