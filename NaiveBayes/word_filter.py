import numpy as np
from functools import reduce
# 屏蔽侮辱性言论

def loadDataSet():
    """
    创建数据集
    :return:
    """

    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签向量，1代表侮辱性词汇，0代表不是
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createvocalist(dataset):
    """
    将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
    :param dataset:整理的数据集
    :return:词汇表
    """
    vocalset = set([])
    for document in dataset:
        vocalset = vocalset | set(document)

    return list(vocalset)

def word2vec(vocalist,inputset):
    """
    根据vocalist词汇表，将inputset向量化，向量的每一个元素为1或0
    :param vocalist: createvocaliset创建的词汇表
    :param inutset:切分的词条列表
    :return:文档向量，词集模型
    """
    returnvec = [0] * len(vocalist)
    print(vocalist)
    for word in inputset:
        if word in vocalist:
            print(word)
            returnvec[vocalist.index(word)] = 1
        else:
            print("the word: %s is not in my vocalist"%word)
        print(returnvec)
    return returnvec

def trainNB(data,label):
    """
    贝叶斯分类器
    :param data:
    :param label:
    :return:侮辱条件概率 非侮辱条件概率 文档属于侮辱概率
    """
    numData = len(data)
    numword = len(data[0])
    # 文档属于侮辱概率
    pAusice = sum(label)/float(numData)
    # 分子
    p0num = np.zeros(numword)
    p1num = np.zeros(numword)
    # 分母
    p0Denom = 0.0
    p1Denom = 0.0

    for i in range(numData):
        if label[i]==1:
            p1num += data[i]
            p1Denom += sum(data[i])
        if label[i]==0:
            p0num += data[i]
            p0Denom += sum(data[i])
    p1vec = p1num/p1Denom
    p0vec = p0num/p0Denom

    return pAusice,p0vec,p1vec

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):

     p1 = reduce(lambda x,y:x*y, vec2Classify * p1Vec) * pClass1     #对应元素相乘
     p0 = reduce(lambda x,y:x*y, vec2Classify * p0Vec) * (1.0 - pClass1)
     print('p0:',p0)
     print('p1:',p1)
     if p1.any() > p0.any():
        return 1
     else:
        return 0



if __name__ == '__main__':
    data,label = loadDataSet()
    # print(data)
    myvocalist = createvocalist(data)
    trainMat = []
    for vec in data:
        trainMat.append(word2vec(myvocalist,vec))
    p0,p1,p2 = trainNB(trainMat,label)
    p0V, p1V, pAb = trainNB(np.array(trainMat), np.array(label))  # 训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']  # 测试样本1
    thisDoc = np.array(word2vec(myvocalist, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']  # 测试样本2

    thisDoc = np.array(word2vec(myvocalist, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')


