import jieba
import os
import random
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

def TextProcessing(filefolder,testsize = 0.2):
    folder_list = os.listdir(filefolder)
    data_list = []
    class_list = []

    # 便利子文件夹
    for folder in folder_list:
        new_folder_path = os.path.join(filefolder,folder)
        files = os.listdir(new_folder_path)

        j = 1
        for file in files:
            if j>100 :
                break
            with open(os.path.join(new_folder_path,file),'r',encoding='utf-8') as f:
                raw = f.read()
            #     返回一个可迭代的迭代器
            word_cut = jieba.cut(raw,cut_all=False)
            word_list = list(word_cut)

            data_list.append(word_list)
            class_list.append(folder)
            j += 1
    data_class_list = list(zip(data_list,class_list))
    random.shuffle(data_class_list)
    index = int(len(data_class_list)*testsize) + 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data,train_class = zip(*train_list)
    test_data,test_class = zip(*test_list)

    # 统计训练集词频
    all_word_dic = {}
    for word_list in train_data:
        for word in word_list:
            if word not in all_word_dic.keys():
                all_word_dic[word] = 0
            all_word_dic[word] += 1

    all_word_tuple_list = sorted(all_word_dic.items(),key=lambda f:f[0],reverse=True)
    all_word_list,all_word_num = zip(*all_word_tuple_list)
    all_word_list = list(all_word_list)
    return all_word_list,train_data,train_class,test_data,test_class
def makewordset(filefoder):
    words_set = set()
    with open(filefoder,'r',encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word)>0:
                words_set.add(word)
    return words_set

def wordsdict(all_words_list,deletN,stopwordsset = set()):
    """
    文本特征选取
    :param all_words_list: 数据集列表
    :param deletN: 删除的N个高频词
    :param stopwordsset: 结束语
    :return: 特征集
    """
    feature_words = []
    n = 1
    for t in range(deletN,len(all_words_list),1):
        if n>1000:
            break
        # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwordsset and 1<len(all_words_list[t])<5:
            feature_words.append(all_words_list[t])
        n+=1
    return feature_words

def TextClassifier(trian_feature_list,test_feature_list,train_class_list,test_class_list):
    nb = MultinomialNB()
    nb.fit(trian_feature_list,train_class_list)
    test_acc = nb.score(test_feature_list,test_class_list)
    return test_acc

def Textfeature(train_data_list,test_data_list,feature_list):
    def textfeature(text,feature_list):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    trian_feature_list = [textfeature(text,feature_list) for text in train_data_list]
    test_feature_list = [textfeature(text,feature_list) for text in test_data_list]
    return trian_feature_list,test_feature_list


if __name__ == '__main__':
    folderpath = 'D:\Project\Machinelearning/NaiveBayes\SinaMew\Sample'
    a,b,c,d,e = TextProcessing(filefolder=folderpath)
    stopset = makewordset('D:\Project\Machinelearning/NaiveBayes\stopwords_cn.txt')

    test_accuracy_list = []
    # deleteNs = range(0, 1000, 20)  # 0 20 40 60 ... 980
    # for deleteN in deleteNs:
    #     feature_words = wordsdict(a, deleteN, stopset)
    #     train_feature_list, test_feature_list = Textfeature(b, d, feature_words)
    #     test_accuracy = TextClassifier(train_feature_list, test_feature_list, c, e)
    #     test_accuracy_list.append(test_accuracy)
    #
    # plt.figure()
    # plt.plot(deleteNs, test_accuracy_list)
    # plt.title('Relationship of deleteNs and test_accuracy')
    # plt.xlabel('deleteNs')
    # plt.ylabel('test_accuracy')
    # plt.show()
    feature_words = wordsdict(a, 450, stopset)
    train_feature_list, test_feature_list = Textfeature(b, d, feature_words)
    test_accuracy = TextClassifier(train_feature_list, test_feature_list, c, e)
    print(test_accuracy)
