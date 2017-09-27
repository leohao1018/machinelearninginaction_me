#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author leo hao
# os windows 7
import numpy as np


def createVocabList(dataSet):
    '''
    将文本转换为数字向量，建立不重复的词汇表
    :param dataSet:
    :return:
    '''
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
            # else:
            #     print("单词“%s”不再验证范围内" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    '''
    朴素贝叶斯分类器训练集
    :param 文档矩阵，每篇文档类别标签所构成的向量:
    :param 文档分类:
    :return:
    '''
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


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    朴素贝叶斯分类函数
    :param vec2Classify:
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    '''
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
