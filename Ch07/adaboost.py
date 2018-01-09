#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author leo hao
# os windows 7
from numpy import *


def loadSimpData():
    dataMat = matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [1.1, 1.2],
        [2., 1.],
        [2.1, 1.2]
    ])
    classLabels = [1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
    return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = 1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 20.0
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            threshVal = rangeMin + float(j) * stepSize
            for inequal in ['lt', 'gt']:
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # print('split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f' % (i, threshVal, inequal, weightedError))
                if weightedError < (minError):
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print("D:", D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # 确保不会发生除0 溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print('classEst: ', classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        # print('aggClassEst: ', aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        # print('total error: ', errorRate, "\n")
        if errorRate == 0.0:
            break
    return weakClassArr


def adaClassify(dataToClass, classifierArr):
    dataMat = mat(dataToClass)
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print('adaClassify aggClassEst: ', aggClassEst)
    return sign(aggClassEst)


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return mat(dataMat), mat(labelMat)


def testHorse(numIt=50):
    dataMat, labelMat = loadDataSet('horseColicTest2.txt')
    classifierArray = adaBoostTrainDS(dataMat, labelMat, numIt)
    testMat, testLabelMat = loadDataSet('horseColicTraining2.txt')
    prediction = adaClassify(testMat, classifierArray)
    dataCount = len(testLabelMat.T)
    errArr = mat(ones((dataCount, 1)))
    errorSum = errArr[prediction != mat(testLabelMat).T].sum()
    print('errorSum: ', errorSum)
    print('errorRate: ', errorSum / dataCount)


if __name__ == '__main__':
    dataMat, classLabels = loadSimpData()
    # 1.test
    # D = mat(ones((7, 1)) / 7)
    # bestStump, minError, bestClassEst = buildStump(dataMat, classLabels, D)
    # print(bestStump)
    # print(minError)
    # print(bestClassEst)
    # 2.test
    # classifierArray = adaBoostTrainDS(dataMat, classLabels, 9)
    # print('classifierArray: ', classifierArray)
    # classLabel = adaClassify([[0, 1.5]], classifierArray)
    # print('classLabel: ', classLabel)
    # 3.test
    testHorse(50)
    pass
