import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_table('testSet.txt',header=None,names=['x','y'])
x = data['x']
y = data['y']

def distance(data,centers):
    # 计算每个点到簇中心的距离
    dist = np.zeros((data.shape[0],centers.shape[0]))
    for i in range(len(data)):
       for j in range(len(centers)):
           # 可优化
           dist[i,j] = np.sqrt(np.sum((data.iloc[i,:]-centers[j])**2))
    return dist


def near_center(data,centers):
    dist =distance(data,centers)
    near_cen = np.argmin(dist,1)
    return near_cen

def kmeans(data,k,item):

    # step1: 随机给定簇中心k个
    centers = np.random.choice(np.arange(-5,5,0.1),(k,2))
    for _ in range(item):
        # step2: 点归属更新，数据点归类到离它最近的簇上
        near_cen = near_center(data, centers)
        # step3: 簇中心更新： 计算每一个簇的重心
        for ci in range(k):
            centers[ci] = data[near_cen==ci].mean()
    return centers,near_cen

center,near_cen = kmeans(data,k=4,item=10)
plt.scatter(x,y,c=near_cen)
plt.scatter(center[:,0],center[:,1],marker='*',s=500)
plt.show()