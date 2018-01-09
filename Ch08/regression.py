#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author leo hao
# os windows 7

from numpy import *


def loadDataSet(fileName):
    fr = open(fileName)
    numFeat = len(fr.readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    '''
    计算最佳拟合直线
    :param xArr:
    :param yArr:
    :return:
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print('this matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    '''
    局部加权线性回归函数
    :param testPoint:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))  # 创建对角矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("this matrix is singular, can not do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def showView(xArr, yArr):
    ws = standRegres(xArr, yArr)
    print(ws)
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * ws
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


def lwlrTestPlot(xArr, yArr):
    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    yHat = lwlrTest(xArr, xArr, yArr, 1.01)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()


def demoTest():
    xArr, yArr = loadDataSet('ex0.txt')
    # ws = standRegres(xArr, yArr)
    # print('ws id ====> ', ws)
    # showView(xArr, yArr)
    lwlrTestPlot(xArr, yArr)


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


def ageTest():
    abX, abY = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print(rssError(abY[0:99], yHat01.T))
    print(rssError(abY[0:99], yHat1.T))
    print(rssError(abY[0:99], yHat10.T))
    print('======================================')
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print(rssError(abY[100:199], yHat01.T))
    print(rssError(abY[100:199], yHat1.T))
    print(rssError(abY[100:199], yHat10.T))


def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("this matrix is singular, can not do inverse")
        return
    ws = denom.I * xMat.T * yMat
    return ws


def ridgeTest(xArr, yArr):
    '''
    岭回归
    :param xArr:
    :param yArr:
    :return:
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def showRidge():
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()


def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)
    inVar = var(inMat, 0)
    inMat = (inMat - inMeans) / inVar
    return inMat


def stageWise(xArr, yArr, eps=0.1, numIt=100):
    '''
    前向逐步回归
    :param xArr:
    :param yArr:
    :param eps:
    :param numIt:
    :return:
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


def stageWiseTest():
    xArr, yArr = loadDataSet('abalone.txt')
    returnMat = stageWise(xArr, yArr, 0.001, 5000)
    print(returnMat)


if __name__ == '__main__':
    # demoTest()
    # ageTest()
    # showRidge()
    stageWiseTest()