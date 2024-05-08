#Written by Oscary Antonio Dotel
#Purpose - Utilize Leave-One-Out model with ADABOOST & GradientBoostClassifier to view accuracy Prediction of ML on Cuticular HydroCarbon
#Dataset size - Augmented Data Large
import pandas as pd
import numpy as np
import matplotlib as plt
#Imported Libraries from Sklearn for ADABOOST/DecisionTree/GRADIENTBOOSTING 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score as acs

import matplotlib.pyplot as plt

#Queen Cuticular Hydrocarbon chemical signiture of 
#---40 Queen Samples CSV file -- Also Original Data
origData = pd.read_csv("napchc13.csv")
#---540 - 40 organic = 500 augmented Queen Samples
augSmallData = pd.read_csv("1newNapchc13.csv")
#---140 - 40 organic = 100 augmented Queen Samples 
augLargeData = pd.read_csv("newNapchc13.csv")
#Function Designed to specify identification 
def reshapeTreatment(x):
        
        lst = []
        for _ in range(len(x)):

            if x[_] == 'C':
                lst.append([0,0,0])
            elif x[_] == 'DP':
                lst.append([1,0,0])
            elif x[_] == 'IM':
                lst.append([0,1,0])
            elif x[_] == 'IMDP':
                lst.append([1,1,0])
            else: # Pryithian
                lst.append([0,0,1]) 
        return lst




#Variable for easily changing FLOAT value for varience testing 
learningRate = 0.02


#Leave-one-out as var to be use methods for splitting data.
loo = LeaveOneOut()

#List to contain predictions from each iteration from LOO from ADABOOST % GradientBoosing
PredABC, PredGBC = list(), list()
trainABC, trainGBC = list(), list()
def LOO(x):
#List to contain predictions from each iteration from LOO from ADABOOST % GradientBoosing
    PredABC, PredGBC = list(), list()
#Extracting Data values from CSV File
    DataY = x[['Treatment']].values
    DataX = x.drop(['Treatment'], axis=1).values
    DataY = DataY.flatten()
 
     #For loop for each iteration - 40 TOTAL
    for trainX, testX in loo.split(x):
    #Splitting Data
        xTrain, xTest = DataX[trainX,:], DataX[testX,:]
        yTrain = DataY.reshape(len(x),1)[trainX,:]
        #Fitting into Each ML Model and prediction. FLATTEN() -- Data Y to reshape for usage
        abc = AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=40,max_depth=20), n_estimators=100, 
                                learning_rate=learningRate, algorithm="SAMME")
        gbc = GradientBoostingClassifier(n_estimators=100,max_depth=20, learning_rate= learningRate)

        abc.fit(xTrain,yTrain.flatten())
        gbc.fit(xTrain,yTrain.flatten())
        modelABC = abc.predict(xTest)
        modelGBC = gbc.predict(xTest)
        abc=reshapeTreatment(modelABC)
        gbc=reshapeTreatment(modelGBC)
        #Assuming 1 label. ---> convert optional label prediction into matrix (numRow*3) ---> finding overlap between all treatment groups. 
        PredGBC.append(modelGBC[0])
        PredABC.append(modelABC[0])
    
    PredGBC = np.array(reshapeTreatment(PredGBC))
    PredABC = np.array(reshapeTreatment(PredABC))
    DataY = np.array(reshapeTreatment(DataY))

#    Converting Accuracy score from List vs. Data Y into %
    accADA = acs(PredABC.flatten(),DataY.flatten())*100
    accGBC = acs(PredGBC.flatten(),DataY.flatten())*100

    print('Accuracy of ADABOOST Leave-One-Out %.2f' %accADA)
    print('Accuracey of Gradient Boosting Classifier Leave-One-Out %.2f' %accGBC)
print("Original Data - 40 Samples")
LOO(origData)
# # print("Small Augmentation Data - 140 Samples")
# # LOO(augSmallData)
# # print("Large Augmentation Data 540 - Samples")
# # LOO(augLargeData)

# Make legend, set axes limits and labels
plt.legend()


plt.show()
