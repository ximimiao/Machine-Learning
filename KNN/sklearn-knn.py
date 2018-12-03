import numpy as np
from sklearn.neighbors import KNeighborsClassifier
def test(triandata,trainlabels,testdata):
    knn = KNeighborsClassifier()
    knn.fit(triandata,trainlabels)
    result = knn.predict(testdata)
    return result
