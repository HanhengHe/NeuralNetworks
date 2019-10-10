# -*- coding: UTF-8 -*-

import numpy as np
from math import exp


#  basic neural network
#  which means only one hidden layer available

#  parameters

# active function
def Sigmoid(X):
    return 1 / (1 + exp(-X))


# low memory require Predictor
class Predictor:
    def __init__(self, labelsName, HLSize, outputSize, IH, IHThreshold, HO, HOThreshold):
        self.labelsName = labelsName
        self.HLSize = HLSize
        self.outputSize = outputSize
        self.IH = IH
        self.IHThreshold = IHThreshold
        self.HO = HO
        self.HOThreshold = HOThreshold

        self.labelsMat = np.mat(np.zeros((self.outputSize, self.outputSize)))
        for i in range(self.outputSize):
            self.labelsMat[i, self.outputSize - 1 - i] = 1

    def predict(self, X):
        #  type check
        if not isinstance(X, list):
            raise NameError('X should be list')

        #  change type
        X = np.mat(X)

        b = np.mat(np.zeros((1, self.HLSize)))
        yCaret = np.zeros((1, self.outputSize))

        for j in range(self.HLSize):
            b[0, j] = Sigmoid(((X * self.IH[:, j]) - self.IHThreshold[0, j]).tolist()[0][0])

        for j in range(self.outputSize):
            yCaret[0, j] = Sigmoid(((b[0, :] * self.HO[:, j]) - self.HOThreshold[0, j]).tolist()[0][0])

        print(yCaret, end=':; ')

        """temp = []

        for i in range(self.outputSize):
            temp.append(((np.abs(yCaret - self.labelsMat[i, :])) *
                         (np.abs(yCaret - self.labelsMat[i, :])).T).tolist()[0][0])

        print(self.labelsName[temp.index(min(temp))])

        return self.labelsName[temp.index(min(temp))]"""

        temp = (np.abs(yCaret - np.ones((1, self.outputSize)))).tolist()[0]
        print(self.labelsName[temp.index(min(temp))])
        return self.labelsName[temp.index(min(temp))]


# both dataList and labelsList should be list
# dataList like [[data00,data01,data02, ...],[data10,data11,data12, ...], ...]
# labelsList like [label_1, label_2, ...]
# learnRate usually in [0.01, 0.8]. an overSize learnRate will cause unstable learning process
# tol is the quit condition of loop
# HLSize means number of hidden layers. -1 allow computer to make a decision itself

class BasicNN:
    def __init__(self, dataList, labelsList, learnRateIH=0.8, learnRateHO=0.8, errorRate=0.05, maxIter=20, alpha=1,
                 HLSize=-1):
        #  type check
        if not isinstance(dataList, list):
            raise NameError('DataList should be list')

        if not isinstance(labelsList, list):
            raise NameError('LabelsList should be list')

        if len(dataList) != len(labelsList):
            raise NameError('len(dataList) not equal to len(labelsList)')

        if not isinstance(HLSize, int):
            raise NameError('NumHL should be int')

        self.dataMat = np.mat(dataList)  # dataset
        self.numData, dataLen = np.shape(self.dataMat)  # record shape of dataset

        # turn labels into 1 and 0
        self.labelNames = list(set(labelsList))  # for remember the meanings of transformed labels
        self.outputSize = len(self.labelNames)
        self.labelsMat = np.mat(np.zeros((self.numData, self.outputSize)))

        self.transferLabelsMat = np.mat(np.zeros((self.outputSize, self.outputSize)))
        for i in range(self.outputSize):
            self.transferLabelsMat[i, i] = 1

        for i in range(len(labelsList)):
            self.labelsMat[i, self.labelNames.index(labelsList[i])] = 1

        print(self.labelNames)
        # print(self.transferLabelsMat)

        # record parameter
        self.learnRate = (learnRateIH, learnRateHO)
        self.errorRate = errorRate
        self.maxIter = maxIter

        # number of input nur
        self.inputSize = dataLen

        #  base on an exist formula
        self.HLSize = int((self.inputSize + self.outputSize) ** 0.5 + alpha) if HLSize == -1 else HLSize

        # init threshold
        self.IHThreshold = np.mat(np.random.random((1, self.HLSize)))
        self.HOThreshold = np.mat(np.random.random((1, self.outputSize)))

        # init IH(input-hiddenLayer) weight matrix and HO(hiddenLayer-output) weight matrix
        # IH:(I*H); HO(H*O)
        self.IH = np.mat(np.random.random((self.inputSize, self.HLSize)))
        self.HO = np.mat(np.random.random((self.HLSize, self.outputSize)))

    #  train should be call after init
    #  since i wanna return a small size predictor
    def train(self):
        # start training
        for _ in range(self.maxIter):
            # calculate the error rate with the hold data set
            # if small enough the quit the loop
            if self.calculateErrorRate() <= self.errorRate:
                break

            # train with every data set
            for i in range(self.numData):

                # get output of hidden layer
                b = np.mat(np.zeros((1, self.HLSize)))
                for j in range(self.HLSize):
                    b[0, j] = Sigmoid(((self.dataMat[i, :] * self.IH[:, j]) - self.IHThreshold[0, j]).tolist()[0][0])

                # get output of output layer
                yCaret = np.mat(np.zeros((1, self.outputSize)))
                for j in range(self.outputSize):
                    yCaret[0, j] = Sigmoid(((b * self.HO[:, j]) - self.HOThreshold[0, j]).tolist()[0][0])

                # calculate g and e defined by watermelon book
                #  g [size:(1, self.outputSize)] and e [size(1, self.HLSize)] should be narray
                g = yCaret.getA() * (np.ones((1, self.outputSize)) - yCaret).getA() * (self.labelsMat[i, :] - yCaret).getA()
                e = b.getA() * (np.ones((1, self.HLSize)) - b).getA() * ((self.HO * np.mat(g).T).T.getA())  # !!

                #  upgrade weight IH
                # print(self.learnRate[0])
                # print(self.dataMat[i, :])
                print(np.mat(e))
                # print(self.learnRate[0] * self.dataMat[i, :].T * np.mat(e))
                self.IH = self.IH + self.learnRate[0] * self.dataMat[i, :].T * np.mat(e)

                #  upgrade weight HO
                # print(self.learnRate[1] * b.T * np.mat(g))
                self.HO = self.HO + self.learnRate[1] * b.T * np.mat(g)  # not sure

                #  upgrade threshold
                # print(self.learnRate[0] * e)
                # print(self.learnRate[1] * g)
                self.IHThreshold = self.IHThreshold - self.learnRate[0] * e
                self.HOThreshold = self.HOThreshold - self.learnRate[1] * g

            # print(self.IH)
            # print(self.HO)
            # print(self.IHThreshold)
            # print(self.HOThreshold)

        return Predictor(self.labelNames, self.HLSize, self.outputSize, self.IH, self.IHThreshold, self.HO,
                         self.HOThreshold)

    def calculateErrorRate(self):
        #  calculate the error rate
        #  base on matrix IH and HO

        errorCounter = 0

        for i in range(self.numData):

            #  get the output of j-th neuron in hidden layer(after active function)
            b = np.mat(np.zeros((1, self.HLSize)))
            for j in range(self.HLSize):
                b[0, j] = Sigmoid(((self.dataMat[i, :] * self.IH[:, j]) - self.IHThreshold[0, j]).tolist()[0][0])

            # get output of output layer
            yCaret = np.mat(np.zeros((1, self.outputSize)))
            for j in range(self.outputSize):
                yCaret[0, j] = Sigmoid(((b * self.HO[:, j]) - self.HOThreshold[0, j]).tolist()[0][0])

            """temp = []

            for k in range(self.outputSize):
                temp.append((np.abs(yCaret - self.transferLabelsMat[k, :]) *
                             np.abs(yCaret - self.transferLabelsMat[k, :]).T).tolist()[0][0])

            if self.transferLabelsMat[temp.index(min(temp))].tolist()[0] != self.labelsMat[i].tolist()[0]:
                errorCounter += 1"""

            temp = (np.abs(yCaret - np.ones((1, self.outputSize)))).tolist()[0]
            if self.transferLabelsMat[temp.index(min(temp))].tolist()[0] != self.labelsMat[i].tolist()[0]:
                errorCounter += 1

        print(errorCounter / self.numData)

        return errorCounter / self.numData
