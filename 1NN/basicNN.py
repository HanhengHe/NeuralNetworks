# -*- coding: UTF-8 -*-

import numpy as np


#  basic neural network
#  which means only one hidden layer available
#  DNN will be finished later

#  parameters

class Predictor:
    def __init__(self):
        pass

    def predict(self, x):
        pass

# both dataList and labelsList should be list
# dataList like [[data00,data01,data02,...],[data10,data11,data12,...],...]
# learnRate usually in [0.01, 0.8]. an overSize learnRate will cause unstable learning process
# tol is the quit condition of loop
# numHL means number of hidden layers. -1 allow computer to make a decision itself

class BasicNN:
    def __init__(self, dataList, labelsList, learnRate=0.8, tol=0.05, maxIter=20, numHL=-1):
        #  type check
        if not isinstance(dataList, list):
            raise NameError('DataList should be list')

        if not isinstance(labelsList, list):
            raise NameError('LabelsList should be list')

        if len(dataList) != len(labelsList):
            raise NameError('len(dataList) not equal to len(labelsList)')

        if not isinstance(numHL, int):
            raise NameError('NumHL should be int')

        #  calculate the number of labels
        self.num = len(labelsList)

        self.inputSize = len(dataList[0][0])

        dataMat = np.mat(dataList)

        for i in range(maxIter):



#  work part
