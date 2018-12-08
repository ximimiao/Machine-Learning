from sklearn.linear_model import LogisticRegression

def colicsklearn():

    frtrain = open('D:\Project\Machinelearning\Logistic\horseColicTraining.txt')
    frtest = open('D:\Project\Machinelearning\Logistic\horseColicTest.txt')
    trainingset = []
    traininglabels = []
    testset = []
    testlabel = []
    for line in frtrain.readlines():
        curline = line.strip().split('\t')
        lineArr = []
        for i in range(len(curline) - 1):
            lineArr.append(float(curline[i]))
        trainingset.append(lineArr)
        traininglabels.append(float(curline[-1]))

    for line in frtest.readlines():
        curline = line.strip().split('\t')
        lineArr = []
        for i in range(len(curline) - 1):
            lineArr.append(float(curline[i]))
        testset.append(lineArr)
        testlabel.append(float(curline[-1]))
    classifier = LogisticRegression(solver='sag',max_iter=1000).fit(trainingset,traininglabels)
    testaccuracy = classifier.score(testset,testlabel)*100
    print("  %f"%testaccuracy)

if __name__ =='__main__':
    colicsklearn()