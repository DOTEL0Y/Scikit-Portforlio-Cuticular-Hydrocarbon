#Written by Oscary Antonio Dotel
#Purpose - Utilize Leave-One-Out model with ADABOOST & GradientBoostClassifier to view accuracy Prediction of ML on Cuticular HydroCarbon
#Dataset size - Small
import pandas as pd
import numpy as np
import matplotlib as plt
#Inported Libraries from Sklearn for ADABOOST&DecisionTree/GRADIENTBOOSTING 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score as acs

#Queen Cuticular Hydrocarbon chemical signiture of 40 Queen Samples 
allData = pd.read_csv("napchc13.csv")

#Extracting Data values from CSV File
dataY = allData[['Treatment']].values
dataX = allData.drop(['Queen','Treatment'], axis=1).values

#Change Data Array so it can be used 
dataY = dataY.flatten()

#Variable for easily changing FLOAT value for varience testing 
learningRate = 0.1

#Model classifier - Learning Rate 0.1
abc = AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=40,max_depth=20), n_estimators=100, learning_rate=learningRate, algorithm="SAMME")
gbc = GradientBoostingClassifier(n_estimators=100,max_depth=20, learning_rate= learningRate)

#Leave-one-out as var to be use methods for splitting data.
loo = LeaveOneOut()

#List to contain predictions from each iteration from LOO from ADABOOST % GradientBoosing
PredABC, PredGBC = list(), list()

#For loop for each iteration - 40 TOTAL
for trainX, testX in loo.split(dataX):
    #Splitting Data
    xTrain, xTest = dataX[trainX,:], dataX[testX,:]
    yTrain, yTest = dataY.reshape(40,1)[trainX,:], dataY.reshape(40,1)[trainX,:]
    #Fitting into Each ML Model and prediction. FLATTEN() -- Data Y to reshape for usage
    abc.fit(xTrain,yTrain.flatten())
    gbc.fit(xTrain,yTrain.flatten())
    modelABC = abc.predict(xTest)
    modelGBC = gbc.predict(xTest)
    #Appending each prediction into List for Latter comparasion
    PredGBC.append(modelGBC[0])
    PredABC.append(modelABC[0])

#Converting Accuracy score from List vs. Data Y into %
accADA = acs(PredABC,dataY.flatten())*100
accGBC = acs(PredGBC,dataY.flatten())*100
print('Accuracy of ADABOOST Leave-Out-One %.2f' % accADA)
print('Accuracey of Gradient Boosting Classifier Leave-Out-One %.2f' %accGBC)
