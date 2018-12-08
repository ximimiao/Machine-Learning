import random
import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

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

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
def testClassifier():
    frtrain = open('D:\Project\Machinelearning\Logistic\horseColicTraining.txt')
    frtest = open('D:\Project\Machinelearning\Logistic\horseColicTest.txt')
    trainingset = []
    traininglabels = []
    for line in frtrain.readlines():
        curline = line.strip().split('\t')
        lineArr = []
        for i in range(len(curline)-1):
            lineArr.append(float(curline[i]))
        trainingset.append(lineArr)
        traininglabels.append(float(curline[-1]))
    trainweights = randGradAscent1(np.array(trainingset),traininglabels,500)
    errorcount = 0
    numTestvec = 0
    for line in frtest.readlines():
        numTestvec += 1.0
        curline = line.strip().split('\t')
        lineArr = []
        for i in range(len(curline)-1):
            lineArr.append(float(curline[i]))
        if int(classifyVector(curline,trainweights)) != int(curline[-1]):
            errorcount +=1
    errorate = float(errorcount)/numTestvec
    print("错误率%f"%errorate)

if __name__ =='__main__':
    testClassifier()

