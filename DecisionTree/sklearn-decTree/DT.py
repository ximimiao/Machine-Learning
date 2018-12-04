import pandas as pd
import sklearn.tree as tree
from sklearn.externals.six import StringIO
from sklearn.preprocessing import LabelEncoder
import pydotplus
if __name__ =='__main__':
    fr = open('D:\Project\Machinelearning\DecisionTree\sklearn-decTree\data.txt')
    lenses = [inst.strip().split(',') for inst in fr.readlines()]
    lenseslabels =  ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])
    lenses_list = []


    lenses_dict = {}

    for each_labels in lenseslabels:
        for each in lenses:
            lenses_list.append(each[each_labels.index(each_labels)])
        lenses_dict[each_labels] = lenses_list
        lenses_list = []
    print(lenses_dict)
    lences_pd =  pd.DataFrame(lenses_dict)
    print(lences_pd)
    le = LabelEncoder()
    for col in  lences_pd.columns:
        lences_pd[col] = le.fit_transform(lences_pd[col])
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf.fit(lences_pd.values.tolist(),lenses_target)
    dot_data = StringIO()
    tree.export_graphviz(clf,out_file=dot_data,feature_names=lences_pd.keys(),
                         class_names=clf.classes_,filled=True,rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")


