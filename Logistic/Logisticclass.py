import numpy as np
import matplotlib.pyplot as plt
import random

def loaddata():
    dataMat = []
    labelMat = []

    fr = open('D:\Project\Machinelearning\Logistic/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,lineArr[0],lineArr[1]])
        labelMat.append(lineArr[2])

    fr.close()
    return  dataMat,labelMat
def plotdataset():
    datamat,labelmat = loaddata()
    dataArr = np.array(datamat)
    n = np.shape(dataArr)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(labelmat[i]) == 1:
            x_cord1.append(dataArr[i,1])
            y_cord1.append(dataArr[i,2])
        else:
            x_cord2.append(dataArr[i,1])
            y_cord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(x_cord1, y_cord1, s=20, c='red', marker='s', alpha=.5)  # 绘制正样本
    ax.scatter(x_cord2,y_cord2, s = 20, c = 'green',alpha=.5)  # 绘制title
    plt.xlabel('x');
    plt.ylabel('y')  # 绘制label
    plt.show()

def sigmoid(x):
    return 1.0/(1+np.exp(-x))
def gradAscent(dataMatIn, classLabels):
    """
    梯度上升算法

    """
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()
def randGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 学习率是随机的
            alpha = 4/(1.0+j+i)+0.01
            # 不选取所有样本，随机选择
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(np.sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
if __name__ == '__main__':
    data,label = loaddata()
    print(randGradAscent1(np.array(data),label))





