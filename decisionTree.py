import math

class Node:
    def __init__(self, label: str, attributes = {}) -> None:
        self.label = label
        self.attributes = attributes
    def predict(self, data):
        if len(self.attributes) == 0:
            return self.label
        reduceData = dict(data)
        del reduceData[self.label]
        res = self.attributes[data[self.label]].predict(reduceData)
        return res
    
class MyDecisionTreeClassifier():
    def __init__(self, title: list, x_train, y_train, criterion='entropy') -> None:
        if len(title) != len(x_train[0]) or len(x_train) != len(y_train):
            raise ValueError('Parameters is not valid!')
        if criterion not in ['entropy', 'gini']:
            raise ValueError('Invalid value for paramater criterion!')
        self.title = title
        self.criterion = criterion
        self.yTrain = y_train
        self.xTrain = []
        if not str(type(x_train[0])) == "<class 'dict'>":
            for i in x_train:
                record = {}
                for index, val in enumerate(i):
                    record[self.title[index]] = str(val)
                self.xTrain.append(record)
        else:
            self.xTrain = x_train

    def entropy(self, data, className, title):
        att = set(map(lambda x: x[title], data))
        Point = 0 if self.criterion == 'entropy' else 1
        for i in att:
            yclass = []
            for index, val in enumerate(data):
                if val[title] == i:
                    yclass.append(className[index])
            numberOfClass = set(yclass)
            totalClass = len(yclass)
            subPoint = 0
            for j in numberOfClass:
                count = yclass.count(j)
                subPoint += -(count/totalClass)*math.log(count/totalClass) if self.criterion == 'entropy' else (count/totalClass)**2
            Point += totalClass/len(data)*subPoint if self.criterion == 'entropy' else -totalClass/len(data)*subPoint
        return Point

    def buildTree(self, xData, yClass, labels: list):
        className = list(set(yClass))
        if len(className) == 1:
            return Node(className[0])
        if len(labels) == 0:
            count = []
            for i in className:
                count.append(yClass.count(i))
            return Node(str(className[count.index(max(count))]))
        
        Point = []
        for label in labels:
            Point.append(self.entropy(xData, yClass, label))
        
        minPointStr = labels.__getitem__(Point.index(min(Point)))
        listAtt = set(map(lambda x: x[minPointStr], xData))
        nodeAtt = {}
        labelsReduce = list(labels)
        labelsReduce.remove(minPointStr)
        for i in listAtt:
            subData = []
            subLabels = []
            for index, val in enumerate(xData):
                if val[minPointStr] == i:
                    subData.append(val)
                    subLabels.append(yClass[index])
            nodeAtt[i] = self.buildTree(subData, subLabels, labelsReduce)
        return Node(minPointStr, nodeAtt)

    def fit(self):
        self.tree = self.buildTree(self.xTrain, self.yTrain, self.title)

    def predict(self, data: list):
        if not str(type(data[0])) == "<class 'dict'>":
            newData = []
            for i in data:
                record = {}
                for index, val in enumerate(i):
                    record[self.title[index]] = val
                newData.append(record)
        else:
            newData = data
        return list(map(lambda x: self.tree.predict(x), newData))