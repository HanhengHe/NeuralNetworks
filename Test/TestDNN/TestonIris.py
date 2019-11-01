from BasicDNN.basicDNN import BasicDNN

#  on iris

filePath = "D:\\WINTER\\Pycharm_project\\data\\Iris"

trainSet = open(filePath+"\\iris")

trainList = []
labelsList = []

for line in trainSet.readlines():
    data = line.split(' ')
    sumUp = float(data[1]) + float(data[2]) + float(data[3]) + float(data[4])
    trainList.append([float(data[1]) / sumUp, float(data[2]) / sumUp, float(data[3]) / sumUp, float(data[4]) / sumUp])
    labelsList.append(data[5].replace('\n', ''))

basicNN = BasicDNN(trainList, labelsList, Depth=1, errorRate=0.5, maxIter=200, learnRateIH=0.8, learnRateH=0.8, learnRateHO=0.8).train()

testSet = open(filePath+"\\iris_test")

testList = []
testLList = []

result = 0
counter = 0

for line in testSet.readlines():
    data = line.split(' ')
    temp = [float(data[1]), float(data[2]), float(data[3]), float(data[4])]
    if basicNN.predict(temp) != data[5].replace('\n', ''):
        counter += 1
    result += 1

print("error rate is %s. " % str(counter/result))
