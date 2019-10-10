# -*- coding: UTF-8 -*-

from BasicNN.basicNN import BasicNN
import numpy as np

####################################################
#                   size parameter                 #
####################################################
# svc get double size
trainSize = 100
testSize = 50

####################################################
#                    parameter                     #
####################################################

trainFilePath = 'D:\\WINTER\\Pycharm_project\\data\\Mnist\\train'
testFilePath = 'D:\\WINTER\\Pycharm_project\\data\\Mnist\\test'

####################################################
trainList = []
trainLabels = []

testList = []
testLabels = []

trainCounter = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [0, 0]]

testCounter = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [0, 0]]

###########################################################################################
#                                      init data                                          #
###########################################################################################

trainFile = open(trainFilePath)

#   assistant data
for line in trainFile.readlines():
    #  修改格式
    dataSet = line.split(')')[0]
    label = line.split(')')[1].replace('\n', '')

    #  ****************从这开始*********************
    if trainCounter[int(label)][1] == trainSize:
        continue

    trainCounter[int(label)][1] = trainCounter[int(label)][1] + 1

    #  ******************到这***********************

    dataSets = dataSet.split(',')
    dataSets[0] = dataSets[0].replace('(', '')

    #  data set调整为float类型，label调整为int类型
    tempIn = []
    for i in range(len(dataSets)):
        t = int(dataSets[i]) / 255  # 归一化
        # 四舍五入
        if t > 0.5:
            tempIn.append(1)
        else:
            tempIn.append(0)

    trainLabels.append(label)

    #  置入数据结构中
    trainList.append(tempIn)

trainFile.close()

testFile = open(testFilePath)

#   data source and test data
for line in testFile.readlines():
    #  修改格式
    dataSet = line.split(')')[0]
    label = line.split(')')[1].replace('\n', '')

    # 计数
    #  ****************从这开始*********************
    if testCounter[int(label)][1] == testSize:
        continue

    testCounter[int(label)][1] = testCounter[int(label)][1] + 1

    #  ******************到这***********************

    dataSets = dataSet.split(',')
    dataSets[0] = dataSets[0].replace('(', '')

    #  data set调整为float类型，label调整为int类型
    tempIn = []
    for i in range(len(dataSets)):

        t = int(dataSets[i]) / 255  # 归一化
        # 四舍五入
        if t > 0.5:
            tempIn.append(1)
        else:
            tempIn.append(0)

    testLabels.append(label)

    #  置入数据结构中
    testList.append(tempIn)

testFile.close()

basicNN = BasicNN(trainList, trainLabels, HLSize=10, errorRate=0.1, maxIter=10000, learnRateIH=0.8, learnRateHO=0.8).train()

############################################################################
#                                 test                                     #
############################################################################

result = 0
counter = 0

for i in range(len(testList)):
    if basicNN.predict(testList[i]) != testLabels[i]:
        counter += 1
    result += 1

print("error rate is %s. " % str(counter / result))
