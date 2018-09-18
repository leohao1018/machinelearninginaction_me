#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author leo hao
# os windows 7
import random

from termcolor import *
import jieba
import pandas as pd
import requests
import time
import numpy as np
from bs4 import BeautifulSoup
# from coverage.files import os

from bayes_function import createVocabList, bagOfWords2VecMN, trainNB0, classifyNB


class goods_classifier:
    '''
    定义分类器
    '''

    def __init__(self):
        self.stopWordsList = self.getStopWords()
        self.carCategories = self.getCarCategory()

    def getStopWords(self):
        '''
        读取停用词文件
        :return:
        '''
        stopWordsList = []
        stopWordsFilePath = '{0}\\category_by_bayes\\stopWords.txt'.format(os.path.dirname(os.getcwd()))
        with open(stopWordsFilePath, encoding='utf8') as stopStrFile:
            lines = stopStrFile.readlines()
            for line in lines:
                stopWordsList.append(line.strip('\n'))
        return stopWordsList

    def getCarCategory(self):
        '''
        获取车辆品牌型号
        :return:
        '''
        allCategory = []
        words = ['A', 'B', 'C', 'D', 'E', 'F', 'G'
            , 'H', 'I', 'J', 'K', 'L', 'M', 'N'
            , 'O', 'P', 'Q', 'R', 'S', 'T'
            , 'U', 'V', 'W', 'X', 'Y', 'Z']
        for word in words:
            url = 'http://www.autohome.com.cn/grade/carhtml/{0}.html'.format(word)

            html = requests.get(url).content.decode('GB2312')
            soup = BeautifulSoup(html, 'lxml')
            categories = soup.select('.rank-list-ul > li > h4 > a')
            for category in categories:
                allCategory.append(category.text)
            time.sleep(10)
        return allCategory

    def getCarGoodsWords(self, goodsName):
        """
        商品名称分词（获取汽车相关词）
        :param goodsName:
        :return:
        """
        stopWords = {}.fromkeys(self.stopWordsList)
        segs = jieba.cut(goodsName, cut_all=False)
        final = []
        for seg in segs:
            if seg not in stopWords and seg.strip() != '' and (
                    '车' in seg or seg in self.carCategories):  # 去除停用词，只获取带‘车’字的词
                final.append(seg)
        return final

    def getGoodsWords(self, goodsName):
        """
        商品名称分词（仅去除停用词）
        :param goodsName:
        :return:
        """
        stopWords = {}.fromkeys(self.stopWordsList)
        segs = jieba.cut(goodsName, cut_all=False)
        final = []
        for seg in segs:
            if seg not in stopWords and seg.strip() != '':  # 去除停用词
                final.append(seg)
        return final

    def toClassifier(self, testNameSet, dataFilePath, observer):
        """
        定义分类器分类
        :return:
        """
        excelDataSet = pd.read_excel(dataFilePath)
        goodsNameDataSet = excelDataSet['goodsName']
        flagSet = excelDataSet['flag']
        import math
        fs = [f for f in flagSet.values if not math.isnan(float(f))]
        trainCount = len(fs)

        docList = []
        fullText = []
        for goodsName in goodsNameDataSet:
            wordList = observer(goodsName)
            docList.append(wordList)
            fullText.extend(wordList)

        # 添加待测试数据词
        for testGoodsName in testNameSet:
            wordList = observer(testGoodsName)
            docList.append(wordList)
            fullText.extend(wordList)

        vocabList = createVocabList(docList)

        # 1. 获取测试集和训练数据集
        trainSet = range(trainCount)
        testCount = 25
        # a. 随机获取测试集，用于计算准确率
        testSet = []
        for docIndex in range(testCount):
            randIndex = int(random.uniform(0, len(trainSet)))
            testSet.append(trainSet[randIndex])
            del (list(trainSet)[randIndex])

        # b. 训练集，获取概率
        trainMat = []
        trainClasses = []
        for docIndex in trainSet:
            trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
            trainClasses.append(int(flagSet[docIndex]))

        # 1.2 计算概率
        p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
        # 1.3 统计错误率
        errorCount = 0
        for docIndex in testSet:
            wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
            probabilityFlag = classifyNB(np.array(wordVector), p0V, p1V, pSpam)
            originFlag = int(flagSet[docIndex])
            if probabilityFlag != originFlag:
                errorCount += 1
                print(colored("classification error ====> " + (
                        goodsNameDataSet[docIndex] + ', cal flag is ====> ' + str(probabilityFlag)), 'red'))
            print((goodsNameDataSet[docIndex]) + ' ====> ' + str(originFlag))
        print(colored('this error rate is ==> ' + str(float(errorCount / len(testSet))), 'red'))

        # # 2. 获取测试数据集
        # for docIndex in range(trainCount, len(goodsNameDataSet)):
        #     testTxt = docList[docIndex]
        #     wordVector = bagOfWords2VecMN(vocabList, testTxt)
        #     print((goodsNameDataSet[docIndex]) + ' ====> ' + str(classifyNB(np.array(wordVector), p0V, p1V, pSpam)))

        # 获取待测试数据类别
        returnFlagSet = []
        for testGoodsName in testNameSet:
            wordVector = bagOfWords2VecMN(vocabList, observer(testGoodsName))
            probabilityFlag = classifyNB(np.array(wordVector), p0V, p1V, pSpam)
            print(testGoodsName + ' ====> ' + str(probabilityFlag))
            returnFlagSet.append(probabilityFlag)
        return returnFlagSet


if __name__ == '__main__':
    classifier = goods_classifier()
    # dataFilePath = '{0}\\category_by_bayes\\goods_data_car.xlsx'.format(os.path.dirname(os.getcwd()))
    # toCategory(dataFilePath, 330)
    ##########
    # dataFilePath = '{0}\\category_by_bayes\\goods_data_credit.xlsx'.format(os.path.dirname(os.getcwd()))
    # testNameSet = []
    # testFilePath = '{0}\\category_by_bayes\\test_credit.txt'.format(os.path.dirname(os.getcwd()))
    # file = open(testFilePath, encoding='utf8')
    # for line in file.readlines():
    #     testNameSet.append(line.strip('\n'))
    #
    # flagSet = classifier.toClassifier(testNameSet, dataFilePath, classifier.getGoodsWords)
    # print(flagSet)

    ###########
    dataFilePath = '{0}\\category_by_bayes\\goods_data_car.xlsx'.format(os.path.dirname(os.getcwd()))
    testNameSet = []
    testFilePath = '{0}\\category_by_bayes\\test_car.txt'.format(os.path.dirname(os.getcwd()))
    file = open(testFilePath, encoding='utf8')
    for line in file.readlines():
        testNameSet.append(line.strip('\n'))

    flagSet = classifier.toClassifier(testNameSet, dataFilePath, classifier.getCarGoodsWords)
    print(flagSet)
