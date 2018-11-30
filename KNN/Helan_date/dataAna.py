import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.lines as lines
from KNN.Classifier import knnClassifier

def file2matrix(filename):
    """
    读取数据
    :param filename: 地址
    :return: data label
    """
    fr = open(filename)
    arrayOlines = fr.readlines()
    number_lines = len(arrayOlines)
    returnMat = np.zeros((number_lines,3))
    classvector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listLine = line.split('\t')
        returnMat[index,:] = listLine[0:3]
        if listLine[-1] == 'didntLike':
            classvector.append(1)
        if listLine[-1] == 'smallDoses':
            classvector.append(2)
        if listLine[-1] == 'largeDoses':
            classvector.append(3)
        index += 1
    return returnMat,classvector

def showdata(datedata,datelabels):
    """
    数据可视化
    :param datedata:
    :param datelabels:
    :return:
    """

    fig,axs = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize=(13,8))

    number_labels = len(datelabels)
    labelscolor = []
    for i in datelabels:
        if i ==1:
            labelscolor.append('black')
        if i ==2:
            labelscolor.append('red')
        if i ==3:
            labelscolor.append('orange')

    axs[0][0].scatter(x=datedata[:,0],y = datedata[:,1],c=labelscolor,s=15,alpha=0.5)
    axs0_titleset = axs[0][0].set_title(u'飞行与游戏')
    axs0_xset = axs[0][0].set_xlabel(u'飞行')
    axs0_yset = axs[0][0].set_ylabel(u'游戏')
    plt.setp(axs0_titleset,size = 9,color = 'red')
    plt.setp(axs0_xset,size = 6,color = 'black')
    plt.setp(axs0_yset,size = 6,color = 'black')

    axs[0][1].scatter(x=datedata[:, 0], y=datedata[:, 2], c=labelscolor, s=15, alpha=0.5)
    axs0_titleset = axs[0][1].set_title(u'飞行与冰淇淋')
    axs0_xset = axs[0][1].set_xlabel(u'飞行')
    axs0_yset = axs[0][1].set_ylabel(u'冰淇淋')
    plt.setp(axs0_titleset, size=9, color='red')
    plt.setp(axs0_xset, size=6, color='black')
    plt.setp(axs0_yset, size=6, color='black')

    axs[1][0].scatter(x=datedata[:, 1], y=datedata[:, 2], c=labelscolor, s=15, alpha=0.5)
    axs0_titleset = axs[1][0].set_title(u'游戏与冰淇淋')
    axs0_xset = axs[1][0].set_xlabel(u'飞行')
    axs0_yset = axs[1][0].set_ylabel(u'游戏')
    plt.setp(axs0_titleset, size=9, color='red')
    plt.setp(axs0_xset, size=6, color='black')
    plt.setp(axs0_yset, size=6, color='black')

    didntLike = lines.Line2D([],[],color='black',marker='.',markersize=6,label ='didntLike')
    smallDoses = lines.Line2D([],[],color='red',marker='.',markersize=6,label ='smallDoses')
    largeDoses = lines.Line2D([],[],color='orange',marker='.',markersize=6,label ='largeDoses')

    axs[0][0].legend(handles = [didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])

    plt.show()

def autonorm(datedata):
    """
    归一化 减去均值 除以 最大之减最小值
    :param datedata:
    :return:
    """
    minvals = datedata.min(0)
    maxvals = datedata.max(0)
    ranges = maxvals - minvals
    num = datedata.shape[0]
    normdata = datedata - np.tile(minvals,(num,1))
    normdata = normdata/np.tile(ranges,(num,1))
    return normdata,minvals,ranges

def datetest():
    filename = 'D:\Project\Machinelearning\KNN\Helan_date\data.txt'
    data,label = file2matrix(filename)

    horatio = 0.1
    normdata,minval,ranges = autonorm(data)
    m = normdata.shape[0]
    num_testvect = int(m*horatio)
    errorCount = 0.0

    for i in range(num_testvect):
        classifierResult = knnClassifier(data[i,:],data[num_testvect:,:],
                                         label[num_testvect:],k=3)
        print("分类结果：%d\t真实结果：%d"%(classifierResult,label[i]))
        if classifierResult != label[i]:
            errorCount+=1
    print("错误率为%f"%(errorCount/num_testvect))





if __name__ =='__main__':
    datetest()
