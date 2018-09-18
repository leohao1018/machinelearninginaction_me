#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author leo hao
# os windows 7
import numpy as np


def createVocabList(data_set):
    """
    将文本转换为数字向量，建立不重复的词汇表
    :param data_set:
    :return:
    """
    vocabSet = set([])  # create empty set
    for document in data_set:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def bagOfWords2VecMN(vocab_list, input_set):
    returnVec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            returnVec[vocab_list.index(word)] += 1
            # else:
            #     print("单词“%s”不再验证范围内" % word)
    return returnVec


def trainNB0(train_matrix, train_category):
    """
    朴素贝叶斯分类器训练集
    :param train_matrix: 文档矩阵，每篇文档类别标签所构成的向量:
    :param train_category: 文档分类:
    :return:
    """
    numTrainDocs = len(train_matrix)  # 文档矩阵的长度
    numWords = len(train_matrix[0])  # 第一个文档的单词个数
    pAbusive = sum(train_category) / float(numTrainDocs)  # 任意文档属于侮辱性文档概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)  # 初始化两个矩阵，长度为numWords，内容值为1
    p0Denom = 2.0
    p1Denom = 2.0  # 初始化概率
    for i in range(numTrainDocs):
        if train_category[i] == 1:
            p1Num += train_matrix[i]
            p1Denom += sum(train_matrix[i])
        else:
            p0Num += train_matrix[i]
            p0Denom += sum(train_matrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2_classify, p0_vec, p1_vec, p_class1):
    """
    朴素贝叶斯分类函数
    :param vec2_classify:
    :param p0_vec:
    :param p1_vec:
    :param p_class1:
    :return:
    """
    p0 = sum(vec2_classify * p0_vec) + np.log(1.0 - p_class1)
    p1 = sum(vec2_classify * p1_vec) + np.log(p_class1)
    if p1 > p0:
        return 1
    else:
        return 0
