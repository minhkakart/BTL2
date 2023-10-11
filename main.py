import pandas
from decisionTree import MyDecisionTreeClassifier
from perceptron import MyPerceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

try:
    raw_data = pandas.read_csv('mushrooms.csv')
except:
    try:
        raw_data = pandas.read_csv('perceptron/mushrooms.csv')
    except:
        raise Exception('File khong ton tai hoac duong dan khong chinh xac!')

## Đọc dữ liệu
dataEndcode = raw_data.apply(LabelEncoder().fit_transform)
## Loại bỏ tiêu đề
title = ['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
data = raw_data[title].values
dataEndcoded = dataEndcode[title].values

trainEncoded, testEncoded = train_test_split(dataEndcoded, test_size=0.3, shuffle=True)
xTrainEndcoded, yTrainEndcoded = trainEncoded[:,1:15].tolist(), trainEncoded[:,0].tolist()
xTestEndcoded, yTestEncoded = testEncoded[:,1:15].tolist(), testEncoded[:,0].tolist()

## Perceptron
perceptron = MyPerceptron(xTrainEndcoded, yTrainEndcoded)
perceptron.fit(eta=0.1)
perceptronPredict = perceptron.predict(xTestEndcoded)
print('Perceptron:')
print('Accuracy score:', accuracy_score(yTestEncoded, perceptronPredict))
print('Precision score:', precision_score(yTestEncoded, perceptronPredict))
print('Recall score:', recall_score(yTestEncoded, perceptronPredict))
print('F1 score:', f1_score(yTestEncoded, perceptronPredict))
print()

## SVM
svm = SVC(kernel='poly', gamma='scale')
svm.fit(xTrainEndcoded, yTrainEndcoded)
svmPredict = svm.predict(xTestEndcoded)
print('SVM:')
print('Accuracy score:', accuracy_score(yTestEncoded, svmPredict))
print('Precision score:', precision_score(yTestEncoded, svmPredict))
print('Recall score:', recall_score(yTestEncoded, svmPredict))
print('F1 score:', f1_score(yTestEncoded, svmPredict))
print()

## Decision tree
train, test = train_test_split(data, test_size=0.3, shuffle=True)
xTrain, yTrain = train[:,1:7].tolist(), train[:,0].tolist()
xTest, yTest = test[:,1:7].tolist(), test[:,0].tolist()
titleForDecision = title[1:7]

## ID3
decisionTreeID3 = MyDecisionTreeClassifier(title=titleForDecision, x_train=xTrain, y_train=yTrain, criterion='entropy')
decisionTreeID3.fit()
decisionTreeID3Predict = decisionTreeID3.predict(xTest)
print('Decision tree ID3 score:')
print('Accuracy score:', accuracy_score(yTest, decisionTreeID3Predict))
print('Precision score:', precision_score(yTest, decisionTreeID3Predict, pos_label='e'))
print('Recall score:', recall_score(yTest, decisionTreeID3Predict, pos_label='e'))
print('F1 score:', f1_score(yTest, decisionTreeID3Predict, pos_label='e'))
print()

## CART
decisionTreeCART = MyDecisionTreeClassifier(title=titleForDecision, x_train=xTrain, y_train=yTrain, criterion='gini')
decisionTreeCART.fit()
decisionTreeCARTPredict = decisionTreeCART.predict(xTest)
print('Decision tree CART score:')
print('Accuracy score:', accuracy_score(yTest, decisionTreeCARTPredict))
print('Precision score:', precision_score(yTest, decisionTreeCARTPredict, pos_label='e'))
print('Recall score:', recall_score(yTest, decisionTreeCARTPredict, pos_label='e'))
print('F1 score:', f1_score(yTest, decisionTreeCARTPredict, pos_label='e'))
print()