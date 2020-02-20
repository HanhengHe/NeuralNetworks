from BasicNN.basicNN import BasicNN

#  on iris

filePath = "D:\\WINTER\\Pycharm_project\\data\\Iris"

trainSet = open(filePath+"\\iris")

trainList = []
labelsList = []

for line in trainSet.readlines():
    data = line.split(' ')
    trainList.append([float(data[1]), float(data[2]), float(data[3]), float(data[4])])
    labelsList.append(data[5].replace('\n', ''))

basicNN = BasicNN(trainList, labelsList, errorRate=0.1, maxIter=50, learnRateIH=0.2, learnRateHO=0.2).train()

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
