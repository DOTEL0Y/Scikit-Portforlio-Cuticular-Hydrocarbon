#Written by Oscary Antonio Dotel
#Purpose - Utilize Leave-One-Out model with ADABOOST & GradientBoostClassifier to view accuracy Prediction of ML on Cuticular HydroCarbon
#Dataset size - Small
import pandas as pd
import numpy as np
import matplotlib as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score as acs

allData = pd.read_csv("napchc13.csv")

dataY = allData[['Treatment']].values
dataX = allData.drop(['Queen','Treatment'], axis=1).values

dataY = dataY.flatten()

abc = AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=40,max_depth=20), n_estimators=100, learning_rate=0.1, algorithm="SAMME")
gbc = GradientBoostingClassifier(n_estimators=100,max_depth=20, learning_rate= 0.1)

loo = LeaveOneOut()
PredABC, PredGBC = list(), list()

for trainX, testX in loo.split(dataX):

    xTrain, xTest = dataX[trainX,:], dataX[testX,:]
    yTrain, yTest = dataY.reshape(40,1)[trainX,:], dataY.reshape(40,1)[trainX,:]
    #print(yTrain.shape)
    abc.fit(xTrain,yTrain.flatten())
    gbc.fit(xTrain,yTrain.flatten())
    modelABC = abc.predict(xTest)
    modelGBC = gbc.predict(xTest)

    PredGBC.append(modelGBC[0])
    PredABC.append(modelABC[0])

accADA = acs(PredABC,dataY.flatten())*100
# accGBC = acs(PredGBC,dataY.flatten())*100
print('Accuracy of ADABOOST Leave-Out-One %.2f' % accADA)
# print('Accuracey of Gradient Boosting Classifier Leave-Out-One %.2f' %accGBC)
