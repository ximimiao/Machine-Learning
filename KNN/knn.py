import numpy as np
import operator

def create_dataset():
    """
    创建数据集
    :return:
    """
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    labels = ['爱情片','爱情片','动作片','动作片']
    return group,labels

def knnClassifier(inx,dataset,labels,k):
    """
    KNN算法 假设给定一个训练数据集，其中实力类别已经确定。分类时，
    对新的实例，根据其k个最近邻的训练示例的类别通过多数表决方式进行预测
    :param data:测试集
    :param dataset:训练集
    :param labels:标签
    :param k:选择距离最小的k个点
    :return:分类结果
    """
    # 返回数据行数
    datasize = dataset.shape[0]
    # 将输入复制datasize行，形成新的数组 减去dataset
    diff = np.tile(inx,(datasize,1)) - dataset
    # 平方
    sqDiff = diff**2
    # 求和
    sqdistance = np.sum(sqDiff,axis=1)
    # 开根号
    distances = sqdistance ** 0.5
    # 对距离进行排序，返回索引
    sortedDistance = np.argsort(distances)
    # 字典记录类别次数
    classCount = {}

    for i in range(k):
        votelabel  = labels[sortedDistance[i]]
        classCount[votelabel] = classCount.get(votelabel,0)+1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    group,labels = create_dataset()
    test = [12,342]
    testclass = knnClassifier(test,group,labels,3)
    print(testclass)


