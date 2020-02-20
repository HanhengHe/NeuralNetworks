# -*- coding: UTF-8 -*-
from BasicDNN.basicDNN import BasicDNN
from BasicNN.basicNN import BasicNN
import numpy as np
from PIL import Image
import cv2

####################################################
#                   size parameter                 #
####################################################
# svc get double size
trainSize = 1200
testSize = 200

####################################################
#                    parameter                     #
####################################################

trainFilePath = 'D:\\workspace\\PycharmProjects\\data\\Mnist\\train'
testFilePath = 'D:\\workspace\\PycharmProjects\\data\\Mnist\\test'

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

#   assistant data 28*28
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

    gray = np.reshape(np.array(dataSets), (28, 28)).astype(np.uint8)
    poolingGary = np.zeros((7, 7))
    for i in range(0, 7):
        for j in range(0, 7):
            poolingGary[i, j] = np.max(gray[i*4:i*4+4, j*4:j*4+4])
    # im = Image.fromarray(poolingGary)  # numpy 转 image类
    # im.show()
    dataSets = np.reshape(poolingGary, (1, 49)).tolist()[0]

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

    gray = np.reshape(np.array(dataSets), (28, 28)).astype(np.uint8)
    poolingGary = np.zeros((7, 7))
    for i in range(0, 7):
        for j in range(0, 7):
            poolingGary[i, j] = np.max(gray[i * 4:i * 4 + 4, j * 4:j * 4 + 4])
    # im = Image.fromarray(poolingGary)  # numpy 转 image类
    # im.show()
    dataSets = np.reshape(poolingGary, (1, 49)).tolist()[0]

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

basicNN = BasicDNN(trainList, trainLabels, errorRate=0.05, Depth=1, maxIter=50).train()
# basicNN = BasicNN(trainList, trainLabels, errorRate=0.05, HLSize=100, maxIter=80).train()

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
