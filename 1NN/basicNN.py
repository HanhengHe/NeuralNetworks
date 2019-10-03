# -*- coding: UTF-8 -*-

import numpy as np
from math import exp


#  basic neural network
#  which means only one hidden layer available
#  DNN will be finished later

#  parameters

# active function
def Sigmoid(X):
    return 1 / (1 + exp(-X))


class Predictor:
    def __init__(self):
        pass

    def predict(self, x):
        pass


# both dataList and labelsList should be list
# dataList like [[data00,data01,data02, ...],[data10,data11,data12, ...], ...]
# labelsList like [label_1, label_2, ...]
# learnRate usually in [0.01, 0.8]. an overSize learnRate will cause unstable learning process
# tol is the quit condition of loop
# HLSize means number of hidden layers. -1 allow computer to make a decision itself

class BasicNN:
    def __init__(self, dataList, labelsList, learnRate=0.8, errorRate=0.05, maxIter=20, HLSize=-1):
        #  type check
        if not isinstance(dataList, list):
            raise NameError('DataList should be list')

        if not isinstance(labelsList, list):
            raise NameError('LabelsList should be list')

        if len(dataList) != len(labelsList):
            raise NameError('len(dataList) not equal to len(labelsList)')

        if not isinstance(HLSize, int):
            raise NameError('NumHL should be int')

        self.dataMat = np.mat(dataList)
        self.numData, dataLen = np.shape(self.dataMat)
        # I'm not sure whether this output size is suitable
        self.labelNames = list(set(labelsList))  # for remember the meanings of transformed labels
        self.outputSize = len(self.labelNames)
        self.labelsMat = np.zeros(self.numData, self.outputSize)

        for i in range(len(labelsList)):
            self.labelsMat[i, self.labelNames.index(labelsList[i])] = 1

        self.learnRate = learnRate
        self.errorRate = errorRate
        self.maxIter = maxIter

        self.inputSize = dataLen

        #  base on an exist formula
        self.HLSize = (self.inputSize * self.outputSize) ** 0.5 if HLSize == -1 else HLSize

        # init IH(input-hiddenLayer) weight matrix and HO(hiddenLayer-output) weight matrix
        # IH:(I*H); HO(H*O)
        self.IH = np.random.random((self.inputSize, self.HLSize))
        self.HO = np.random.random((self.HLSize, self.outputSize))

        self.yCaret = np.zeros((1, self.outputSize))

    #  train should be call after init
    #  since i wanna return a small size predictor
    def train(self):
        # start training
        for i in range(self.maxIter):
            if self.calculateErrorRate() <= self.errorRate:
                break

            for j in range(self.numData):
                pass

    def calculateErrorRate(self):
        #  calculate the error rate
        #  base on matrix IH and HO

        errorCounter = 0

        for i in range(self.numData):
            tempMatrix = np.zeros(1, self.HLSize)
            #  get the output of j-th neuron in hidden layer(after active function)
            for j in range(self.HLSize):
                tempMatrix[0, j] = Sigmoid((self.dataMat[i, :] * self.IH[:, j].T).tolist()[0])
            tempOutputMatrix = np.zeros(1, self.outputSize)
            #  get the output of j-th neuron in output(after active function)
            for j in range(self.outputSize):
                tempOutputMatrix[0, j] = Sigmoid((tempMatrix * self.HO[:, j].T).tolist[0])
            temp = tempOutputMatrix - self.labelsMat[i, :]
            errorCounter += (temp * temp.T).tolist()[0]
        return errorCounter / self.numData  # is it right?
#  work part
