from math import log
import operator
import pickle

def createdataset():
    """
    创建数据集
    :return:
    """
    dataSet = [[0, 0, 0, 0, 'no'],
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
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataSet, labels

def calEntrioy(dataset):
    """
    计算给定数据集的经验熵
    """
    numlen = len(dataset)
    labelCounts = {}
    for featvec in dataset:
        currentlabel = featvec[-1]
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel] = 0
        labelCounts[currentlabel] +=1
    Ent = 0.0
    for key in labelCounts:
        prob =  float(labelCounts[key])/numlen
        Ent -= prob*log(prob,2)
    return Ent
def splitdata(dataset,axis,value):
    """
    根据特征划分数据集
    """
    retdataset = []
    for featvec in dataset:
        if featvec[axis] == value:
            reducedfeatvec = featvec[:axis]
            reducedfeatvec.extend(featvec[axis+1:])
            retdataset.append(reducedfeatvec)
    return  retdataset


def calgain(dataset):
    """
    根据信息增益选择最好的特征
    """
    num_feature = len(dataset[0]) - 1
    ent = calEntrioy(dataset)
    bestInfoGain = 0.0
    bestfeat = -1
    for i in range(num_feature):
        # 获取特征i的所有取值
        fealist = [example[i] for example in dataset]
        uniqueVals= set(fealist)
        newEnt = 0.0
        for value in uniqueVals:
            subdataset = splitdata(dataset,i,value)
            prob = len(subdataset)/float(len(dataset))
            newEnt += prob*calEntrioy(subdataset)
        InfoGain = ent - newEnt
        if (InfoGain > bestInfoGain):
            bestInfoGain = InfoGain
            bestfeat = i
    return bestfeat

def majorityCnt(classlist):
    """
    统计class list 出现次数最多的元素
    :param classlist:
    :return:
    """
    classCount = {}
    for vec in classCount:
        if vec not in classCount:
            classCount[vec] = 0
        classCount[vec] += 1
    sortedclasscount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

def createTree(dataset,labels,featlabels):

    """
    ID3算法
    创建决策树
    :param dataset:数据集
    :param labels: 分类属性标签
    :param featlabels:存储选择的最优标签特征
    :return:
    """
    classlist = [example[-1] for example in dataset]
    # 类别相同，停止划分
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    # 遍历完所有特征时，返回出现次数最多的
    if len(classlist)==1 or len(labels) == 0:
        majorityCnt(classlist)
    #选择最优特征索引
    bestFeat = calgain(dataset)
    # 最优标签
    bestFeatLabels = labels[bestFeat]
    featlabels.append(bestFeatLabels)
    Mytree = {bestFeatLabels:{}}
    # 删除已经使用标签
    del(labels[bestFeat])
    # 得到训练集中所有最优特征的值
    featvalues = [example[bestFeat] for example in dataset]
    # 去掉重复值
    uniquefeat = set(featvalues)
    #遍历特征，创建决策树
    for vec in uniquefeat:
        Mytree[bestFeatLabels][vec] = createTree(splitdata(dataset,bestFeat,value=vec),
                                                 labels,featlabels)
    return Mytree



def classify(inputTree,featlabels,testvec):
    """
    使用决策树分类
    """
    # 获取决策树结点
    firstStr = next(iter(inputTree))
    print(firstStr)
    # 下一个字典
    secondDict = inputTree[firstStr]
    featIndex = featlabels.index(firstStr)
    for key in secondDict.keys():
        if testvec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featlabels, testvec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
    """
    存储决策树
    :param inputTree:
    :param filename:
    :return:
    """
    with open(filename,'wb') as fw:
        pickle.dump(inputTree,fw)

def getTree(filename):
    fr = open(filename,'rb')
    return pickle.load(fr)




if __name__ =='__main__':
    data,label = createdataset()
    featlabels= []
    tree = getTree('tree.txt')
    print(tree)