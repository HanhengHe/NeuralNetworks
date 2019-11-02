# -*- coding: UTF-8 -*-

import numpy as np

alpha = 0.3  # alpha for PReLU


#  BEFORE CODE:
#  this is a deep neural network
#  default depth (not include input and output layer) is one
#  and i finally realize that my input neurons have no threshold (but that's not a problem)

# ***********************active function************************
# Sigmoid is used for output
def Sigmoid(X):
    return 1 / (1 + np.exp(-X))


# ReLU for rest
def PReLU(X):
    _, n = np.shape(X)
    for i in range(n):
        X[0, i] = X[0, i] if X[0, i] >= 0 else X[0, i] * alpha
        # X[0, i] = X[0, i] if X[0, i] <= 1 else 1.01
    return X


# d(ReLU)/dx
def dPReLU(X):
    _, n = np.shape(X)
    for i in range(n):
        # X[0, i] = 1 if X[0, i] >= 0 and X[0, i] != 1.01 else 0
        X[0, i] = 1 if X[0, i] >= 0 else alpha
    return X


# **************************************************************


# ************************Predictor*****************************
# low memory require Predictor
class Predictor:
    def __init__(self, labelsName, outputSize, Weight, Threshold):
        self.labelsName = labelsName
        self.outputSize = outputSize
        self.Weight = Weight
        self.Threshold = Threshold

        self.labelsMat = np.mat(np.zeros((self.outputSize, self.outputSize)))
        for i in range(self.outputSize):
            self.labelsMat[i, self.outputSize - 1 - i] = 1

    def predict(self, X):
        #  type check
        if not isinstance(X, list):
            raise NameError('X should be list')

        #  change type
        X = np.mat(X)

        signal = X

        for i in range(0, len(self.Weight) - 1):
            signal = PReLU(signal * self.Weight[i] - self.Threshold[i])

        output = Sigmoid(signal * self.Weight[len(self.Weight) - 1] - self.Threshold[len(self.Threshold) - 1])

        temp = (np.abs(output - np.mat(np.ones((1, self.outputSize))))).tolist()[0]

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

class BasicDNN:
    def __init__(self, dataList, labelsList, Depth=1, learnRateIH=0.8, learnRateH=0.8, learnRateHO=0.8, errorRate=0.05,
                 maxIter=20, alpha=1, HLSize=-1, fixParameter=-1, eta=10**(-5)):
        #  type check
        if not isinstance(dataList, list):
            raise NameError('DataList should be list')

        if not isinstance(labelsList, list):
            raise NameError('LabelsList should be list')

        if len(dataList) != len(labelsList):
            raise NameError('len(dataList) not equal to len(labelsList)')

        if not isinstance(HLSize, int):
            raise NameError('NumHL should be int')

        self.eta = eta

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
        self.learnRate = (learnRateIH, learnRateH, learnRateHO)
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
        self.Weight = [np.mat(np.random.random((self.inputSize, self.HLSize))) * fixParameter]
        self.Threshold = [np.mat(np.random.random((1, self.HLSize))) * fixParameter]

        # H(weight inside hidden layer) weight matrix
        for i in range(Depth - 1):
            self.Weight.append(np.mat(np.random.random((self.HLSize, self.HLSize))) * fixParameter)
            self.Threshold.append(np.mat(np.random.random((1, self.HLSize))) * fixParameter)

        # HO(hiddenLayer-output) weight matrix
        self.Weight.append(np.mat(np.random.random((self.HLSize, self.outputSize))) * fixParameter)
        self.Threshold.append(np.mat(np.random.random((1, self.outputSize))) * fixParameter)

    #  train should be call after init
    #  since i wanna return a small size predictor
    def train(self):
        for s in range(self.maxIter):
            # Gather loss
            # if small enough the quit the loop
            print("step %s, error rate: " %s, end='')
            if self.calculateErrorRate() <= self.errorRate:
                for w in self.Weight:
                    print(w)
                print()
                print(self.Threshold)
                break

            # train with every data set
            for i in range(self.numData):

                currentLabel = self.labelsMat[i, :]

                # Gather output of every neurons
                signal = self.dataMat[i, :]
                Signal = [signal]

                for j in range(0, len(self.Weight) - 1):
                    signal = PReLU(signal * self.Weight[j] - self.Threshold[j])  # signal => z
                    Signal.append(signal)

                # ! DELTA IS REVERSED !
                Delta = []

                # Gather delta for every layer

                # the output-hidden layer (since active function is different)
                lastPotential = Sigmoid(
                    signal * self.Weight[len(self.Weight) - 1] - self.Threshold[len(self.Threshold) - 1])
                delta = lastPotential - currentLabel
                Delta.append(delta.getA() * lastPotential.getA() * (1 - lastPotential).getA())

                # the rest
                for j in range(self.Depth):
                    delta = Delta[j] * self.Weight[len(self.Weight) - j - 1].T
                    """lastInput = Signal[len(Signal) - j - 1] * self.Weight[len(self.Weight) - j - 1] - self.Threshold[
                        len(self.Threshold) - j - 1]"""
                    Delta.append(delta.getA() * (dPReLU(Signal[len(Signal) - j - 1]).getA()))

                # print(Delta)

                temp = 0
                for it in Delta:
                    x, y = np.shape(it)
                    temp += np.sum(np.abs(it)) / (x * y)

                temp = temp / len(Delta)

                if temp <= self.eta:
                    print("Low gradient!")
                    break

                # update Weight and Threshold
                self.Weight[0] -= self.learnRate[0] * Signal[0].T * Delta[len(Delta) - 1]
                self.Threshold[0] += self.learnRate[0] * Delta[len(Delta) - 1]

                self.Weight[len(self.Weight) - 1] -= self.learnRate[2] * Signal[len(Signal) - 1].T * Delta[0]
                self.Threshold[len(self.Weight) - 1] += self.learnRate[2] * Delta[0]

                for j in range(1, len(self.Weight) - 1):
                    self.Weight[j] -= self.learnRate[1] * Signal[j].T * Delta[len(self.Weight) - j - 1]
                    self.Threshold[j] += self.learnRate[1] * Delta[len(self.Weight) - j - 1]

        return Predictor(self.labelNames, self.outputSize, self.Weight, self.Threshold)

    def calculateErrorRate(self):
        #  calculate the error rate

        errorCounter = 0

        for i in range(self.numData):

            signal = self.dataMat[i, :]

            for j in range(len(self.Weight) - 1):
                signal = PReLU(signal * self.Weight[j] - self.Threshold[j])

            output = Sigmoid(signal * self.Weight[len(self.Weight) - 1] - self.Threshold[len(self.Threshold) - 1])

            temp = (np.abs(output - np.mat(np.ones((1, self.outputSize))))).tolist()[0]
            if self.transferLabelsMat[temp.index(min(temp))].tolist()[0] != self.labelsMat[i].tolist()[0]:
                errorCounter += 1

        print(errorCounter / self.numData)

        return errorCounter / self.numData
