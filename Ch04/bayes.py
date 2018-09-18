'''
Created on Oct 19, 2010

@author: Peter
'''
import random

import numpy as np
import re
import feedparser


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


# 1、将文本转换为数字向量
# a、建立不重复的词汇表
def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


# 朴素贝叶斯分类器训练集
def trainNB0(trainMatrix, trainCategory):  # 传入参数为文档矩阵，每篇文档类别标签所构成的向量
    numTrainDocs = len(trainMatrix)  # 文档矩阵的长度
    numWords = len(trainMatrix[0])  # 第一个文档的单词个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 任意文档属于侮辱性文档概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)  # 初始化两个矩阵，长度为numWords，内容值为1
    p0Denom = 2.0
    p1Denom = 2.0  # 初始化概率
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# b、将每条言语转换为数字向量：建立与词汇表同等大小的言语向量，若言语中的词汇在词汇表中出现则标记为1，否则为0.
# vocabList:单词字典集合
# inputSet:单条文本
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("单词“%s”不再验证范围内" % word)
    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
            # else:
            #     print("单词“%s”不再验证范围内" % word)
    return returnVec


# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testTrain():
    listOPosts, listClasses = loadDataSet()  # 产生文档矩阵和对应的标签
    myVocabList = createVocabList(listOPosts)  # 创建并集

    trainMat = []  # 创建一个空的列表
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  # 使用词向量来填充trainMat列表
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))  # 训练函数
    print(p0V)
    print(p1V)
    print(pAb)

    testEntry = ["love", "my", "dalmation"]
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ["AA", "garbage"]
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))


def testEmail():
    mySent = 'This is the best book on Python or M.L I have ever laid eyes upon.'
    regEx = re.compile('\\W+')
    listOfTokens = regEx.split(mySent)
    returnTokens = [tok.lower() for tok in listOfTokens if len(tok) > 0]
    print(returnTokens)
    return returnTokens


def textParse(bigString):
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 0]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)

    # 划分测试集 和 训练集
    trainSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del (list(trainSet)[randIndex])

    # 朴素贝叶斯计算概率
    trainMat = []
    trainClasses = []
    for docIndex in trainSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))

    # 统计错误率
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('this error rate is ==> ', float(errorCount / len(testSet)))


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary

    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])

    trainingSet = range(2 * minLen)
    testSet = []  # create test set
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (list(trainingSet)[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))

    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print('SF*******************************************************************************')
    for item in sortedSF:
        print(item[0])
    sortedSY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print('NY*******************************************************************************')
    for item in sortedSY:
        print(item[0])


if __name__ == '__main__':
    # testTrain()
    # testEmail()
    spamTest()
    # ny = feedparser.parse('https://newyork.craigslist.org/search/ppp?format=rss')
    # sf = feedparser.parse('https://cnj.craigslist.org/search/ppp?format=rss')
    # # vocabList, pSF, pNY = localWords(ny, sf)
    # # vocabList, pSF, pNY = localWords(ny, sf)
    # getTopWords(ny, sf)
