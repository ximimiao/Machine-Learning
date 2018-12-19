import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.grid_search import GridSearchCV

def loadDataSet(fileName):
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat
if __name__ == '__main__':
    dataArr, classLabels = loadDataSet('D:\Project\Machinelearning\Logistic\horseColicTraining.txt')
    testArr, testLabelArr = loadDataSet('D:\Project\Machinelearning\Logistic\horseColicTest.txt')

    # 调参
    param_test1 = {'n_estimators': list(range(20, 81, 10))} # n_estimators': 40
    # min_samples_split': 10, 'max_depth': 11
    param_test2 = {'max_depth': list(range(3, 14, 2)), 'min_samples_split': list(range(10, 801, 200))}
    # 'min_samples_leaf': 16, 'min_samples_split': 10
    param_test3 = {'min_samples_split': list(range(10, 1900, 200)), 'min_samples_leaf': list(range(6, 101, 10))}

    # grid = GridSearchCV(estimator=GradientBoostingClassifier(
    #     learning_rate=0.1,n_estimators=40,max_depth=11,subsample=0.8,max_features='sqrt',
    #     random_state=10),param_grid=param_test3,scoring='roc_auc',
    #     iid=False,cv=5)
    # grid.fit(dataArr, classLabels)
    # print(grid.grid_scores_,grid.best_params_,grid.best_score_)

    bdt = GradientBoostingClassifier(learning_rate=0.1, n_estimators=40, max_depth=11,
                                     min_samples_split=10,min_samples_leaf=16,
                                     subsample=0.8, max_features='sqrt', random_state=10)
    bdt.fit(dataArr, classLabels)
    predictions = bdt.predict(dataArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != classLabels].sum() / len(dataArr) * 100))
    predictions = bdt.predict(testArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != testLabelArr].sum() / len(testArr) * 100))