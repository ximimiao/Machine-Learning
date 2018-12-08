import numpy as np
import matplotlib.pyplot as plt

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
    return 1.0/(1+np.exp(-int(x)))
def gradAscent(dataMatIn, classLabels):
    #转换成numpy的mat
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

if __name__ == '__main__':
    plotdataset()






