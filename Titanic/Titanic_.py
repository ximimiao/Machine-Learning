import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# 读取训练数据
data_train = pd.read_csv('D:\Project\Machinelearning\Titanic\/train.csv')
data_test =  pd.read_csv('D:\Project\Machinelearning\Titanic\/test.csv')
# 数据预处理
# 填充Age缺失值
data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())

# Sex： male = 0 female =1
data_train.loc[data_train['Sex'] == "male","Sex"] = 0
data_train.loc[data_train['Sex'] == "female","Sex"] =1

# Embarked : 先填缺失值 S=0 C=1 Q=2
data_train['Embarked'] = data_train['Embarked'].fillna('S')
data_train.loc[data_train['Embarked'] == "S","Embarked"] = 0
data_train.loc[data_train['Embarked'] == "C","Embarked"] = 1
data_train.loc[data_train['Embarked'] == "Q","Embarked"] = 2


# 选择特征
predictors = ['Pclass','Sex','Age','SibSp','Fare','Embarked']

# 线性回归
alg = LinearRegression()
kf = KFold( n_splits=3,shuffle=True,random_state=1)

# 选择特征
predictors = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

predictions = []

for train_index,test_index in kf.split(data_train):

    train_predictor = (data_train[predictors].iloc[train_index,:])
    train_target = (data_train['Survived'].iloc[train_index])

    alg.fit(train_predictor,train_target)

    test_prediction = alg.predict(data_train[predictors].iloc[test_index,:])

data_test['Age'] = data_test['Age'].fillna(data_test['Age'].median())

# Sex： male = 0 female =1
data_test.loc[data_test['Sex'] == "male","Sex"] = 0
data_test.loc[data_test['Sex'] == "female","Sex"] =1

# Embarked : 先填缺失值 S=0 C=1 Q=2
data_test['Embarked'] = data_test['Embarked'].fillna('S')
data_test.loc[data_test['Embarked'] == "S","Embarked"] = 0
data_test.loc[data_test['Embarked'] == "C","Embarked"] = 1
data_test.loc[data_test['Embarked'] == "Q","Embarked"] = 2

prediction = alg.predict(data_test[predictors])

prediction.to_csv('1.csv')
# result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':prediction.astype(np.int32)})
# result.to_csv("logistic_regression_predictions.csv", index=False)

# predictions = np.concatenate(predictions,axis = 0)
# predictions[predictions >=0.5] = 1
# predictions[predictions <0.5] = 0
# accuracy = np.sum(predictions == data_train["Survived"])/len(predictions)
# print(accuracy)




