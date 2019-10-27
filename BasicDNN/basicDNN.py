# -*- coding: UTF-8 -*-

import numpy as np
from math import exp


#  BEFORE CODE:
#  this is a deep neural network
#  default depth (not include input and output layer) is one
#  and i finally realize that my input neurons have no threshold (but that's not a problem)

# ***********************active function************************
# Sigmoid is used for output
def Sigmoid(X):
    return 1 / (1 + np.exp(-X))


# ReLU for rest
def ReLU(X):
    _, n = np.shape(X)
    for i in range(n):
        X[0, i] = X[0, i] if X[0, i] >= 0 else 0
    return X

# d(ReLU)/dx
def dReLU(x):
    return 1 if x >= 0 else 0

# **************************************************************


# ************************Predictor*****************************
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

        temp = (np.abs(yCaret - np.ones((1, self.outputSize)))).tolist()[0]
        print(self.labelsName[temp.index(min(temp))])
        return self.labelsName[temp.index(min(temp))]


# **************************************************************


# ######################## MAIN CODE ############################

# both dataList and labelsList should be list
# dataList like [[data00,data01,data02, ...],[data10,data11,data12, ...], ...]
# labelsList like [label_1, label_2, ...]
# default depth is one
# learnRate usually in [0.01, 0.8]. an overSize learnRate will cause unstable learning process
# tol is the quit condition of loop
# HLSize means number of hidden layers. -1 allow computer to make a decision itself

class BasicNN:
    def __init__(self, dataList, labelsList, Depth=1, learnRateIH=0.8, learnRateHO=0.8, errorRate=0.05, maxIter=20,
                 alpha=1, HLSize=-1, fixParameter=-1, ):
        #  type check
        if not isinstance(dataList, list):
            raise NameError('DataList should be list')

        if not isinstance(labelsList, list):
            raise NameError('LabelsList should be list')

        if len(dataList) != len(labelsList):
            raise NameError('len(dataList) not equal to len(labelsList)')

        if not isinstance(HLSize, int):
            raise NameError('NumHL should be int')

        self.dataMat = np.mat(dataList)  # dataSet
        self.numData, dataLen = np.shape(self.dataMat)  # record shape of dataSet

        # turn labels into 1 and 0
        self.labelNames = list(set(labelsList))  # for remember the meanings of transformed labels

        # number of output neurons
        self.outputSize = len(self.labelNames)
        self.labelsMat = np.mat(np.zeros((self.numData, self.outputSize)))

        self.transferLabelsMat = np.mat(np.zeros((self.outputSize, self.outputSize)))
        for i in range(self.outputSize):
            self.transferLabelsMat[i, i] = 1

        for i in range(len(labelsList)):
            self.labelsMat[i, self.labelNames.index(labelsList[i])] = 1

        print(self.labelNames)

        # record parameter
        self.learnRate = (learnRateIH, learnRateHO)
        self.errorRate = errorRate
        self.maxIter = maxIter

        # number of input neurons
        self.inputSize = dataLen

        # base on an exist formula
        # number of neurons for each hidden layer
        self.HLSize = int((self.inputSize + self.outputSize) ** 0.5 + alpha) if HLSize == -1 else HLSize

        # depth of neural network (not include input and output layer)
        self.Depth = Depth

        # a part for fix parameter
        # avoid over size weight when training start
        if fixParameter == -1:
            temp = 0
            for i in range(len(dataList)):
                temp += np.sum(self.dataMat[i, :])

            temp = temp / len(dataList)
            fixParameter = 1 / temp

        # weight matrix and Threshold
        # IH:(I*H); H(HLSize*HLSize); HO(H*O)
        # init IH(input-hiddenLayer) weight matrix
        self.Weight = [].append(np.mat(np.random.random((self.inputSize, self.HLSize))) * fixParameter)
        self.Threshold = []

        # H(weight inside hidden layer) weight matrix
        for i in range(Depth - 1):
            self.Weight.append(np.mat(np.random.random((self.HLSize, self.HLSize))) * fixParameter)
            self.Threshold.append(np.mat(np.random.random((1, self.HLSize))))

        # HO(hiddenLayer-output) weight matrix
        self.outputWeight = np.mat(np.random.random((self.HLSize, self.outputSize))) * fixParameter
        self.outputThreshold = np.mat(np.random.random((1, self.outputSize)))

    #  train should be call after init
    #  since i wanna return a small size predictor
    def train(self):
        for _ in range(self.maxIter):
            # Gather loss
            # if small enough the quit the loop
            if self.calculateErrorRate() <= self.errorRate:
                break

            # train with every data set
            for i in range(self.numData):

                # Gather output of every neurons
                signal = self.dataMat[i, :] * self.Weight[0]
                Output = [].append(signal)

                for j in range(1, len(self.Weight)):
                    signal = ReLU(signal * self.Weight[j] - self.Threshold[j - 1])
                    Output.append(signal)

                # actually the last one of Output is useless = =

                Delta = []
                # Gather delta for every layer
                # the output-hidden layer since active function is different
                delta =
                Delta.append()

                # print(b)

                # calculate g and e defined by watermelon book
                #  g [size:(1, self.outputSize)] and e [size(1, self.HLSize)] should be narray
                g = yCaret.getA() * (np.ones((1, self.outputSize)) - yCaret).getA() * (
                        self.labelsMat[i, :] - yCaret).getA()
                e = b.getA() * (np.ones((1, self.HLSize)) - b).getA() * ((self.HO * np.mat(g).T).T.getA())  # !!

                #  upgrade weight IH
                self.IH = self.IH + self.learnRate[0] * self.dataMat[i, :].T * np.mat(e)

                #  upgrade weight HO
                self.HO = self.HO + self.learnRate[1] * b.T * np.mat(g)  # not sure

                #  upgrade threshold
                self.IHThreshold = self.IHThreshold - self.learnRate[0] * e
                self.HOThreshold = self.HOThreshold - self.learnRate[1] * g

        return Predictor(self.labelNames, self.HLSize, self.outputSize, self.IH, self.IHThreshold, self.HO,
                         self.HOThreshold)

    def calculateErrorRate(self):
        #  calculate the error rate

        errorCounter = 0

        for i in range(self.numData):

            signal = self.dataMat[i, :] * self.Weight[0]

            for j in range(1, len(self.Weight)):
                signal = ReLU(signal * self.Weight[j] - self.Threshold[j-1])

            yCaret = Sigmoid(signal * self.outputWeight - self.outputThreshold)

            temp = (np.abs(yCaret - np.mat(np.ones((1, self.outputSize))))).tolist()[0]
            if self.transferLabelsMat[temp.index(min(temp))].tolist()[0] != self.labelsMat[i].tolist()[0]:
                errorCounter += 1

        print(errorCounter / self.numData)

        return errorCounter / self.numData
