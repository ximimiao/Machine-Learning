import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = load_iris()
data = iris.data
target = iris.target
train_data,test_data,train_target,test_target = train_test_split(data,target,test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(train_data,train_target)
print(knn.predict(test_data))
print(test_target)
print(knn.score(test_data,test_target))


