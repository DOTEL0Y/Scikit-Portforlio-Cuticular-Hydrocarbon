#Establish Dataframe
import pandas as pd
#Plotting Data 
import matplotlib.pyplot as plt

#Number Py library to manipulate data
import numpy as np

#ADABOOST algorithm implementation
from sklearn.ensemble import AdaBoostClassifier
#ADABOOST Regressor - singular Training for Chemical molecule
from sklearn.ensemble import AdaBoostRegressor
#Splits Data from Train/Test Data
from sklearn.model_selection import train_test_split

#Establish STUMP for Classifier WEIGHT recalibration 
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



#ADAboost ---> 100 weak learners 
ada = AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=40,max_depth=24),n_estimators=100, algorithm='SAMME',
                                 learning_rate=.02)
#DecisionTreeClassifier -- MAX 40 leaf Nodes
dtc = DecisionTreeClassifier(max_leaf_nodes=40,max_depth=24)

carboClassifier = AdaBoostClassifier(n_estimators=25, algorithm="SAMME")

#csv File into allData Variable  ---> Later modify for TRAINING
allData = pd.read_csv("napchc13.csv")

#Input 'data' var for spliting to TRAINING 
data = allData

#Labels of the Treatment
dataY = data[['Treatment']]

#Data from Curticular Hydrocarbon
dataX = data.drop(["Queen","Treatment"], axis=1).values

chc23_1 = data[["c23:1"]]

# Split dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.5) # 50% training and 50% test
ctest, ctrain = train_test_split(chc23_1, test_size=0.5)
#convert Dataframe to NUM Array
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()



#train = train.reshape(1,20) // ALSO ---> Trains ML Algor
ada.fit(X_train,Y_train.flatten())
dtc.fit(X_train,Y_train.flatten())
carboClassifier.fit(ctrain, Y_train.flatten())

x1 = ada.predict(X_test)
y1 = dtc.predict(X_test)
x2 = carboClassifier.predict(ctest)

accuracyX1 = accuracy_score(Y_test.flatten(), x1)
accuracyY1 = accuracy_score(Y_test.flatten(), y1)
accuracyX2 = accuracy_score(Y_test.flatten(), x2)
#3D Figure 
ax = plt.axes(projection="3d")

ax.set_xlim(0,4)
ax.set_xlabel("Predicted Treatment")

ax.set_ylim(0,1)
ax.set_ylabel("Accuracy Score")

ax.set_zlabel("Actual Treatment")
ax.set_zlim(0,4)

#Converts String to INT for plotting 
def reshapeTreatment(x):
        
        lst = []
        for _ in range(len(x1)):

            if x[_] == 'C':
                lst.append([0])
            elif x[_] == 'DP':
                lst.append([1])
            elif x[_] == 'IM':
                lst.append([2])
            elif x[_] == 'IMDP':
                lst.append([3])
            else:
                lst.append([4]) 
        return lst


ax.scatter(reshapeTreatment(x1),accuracyX1,reshapeTreatment(Y_test.flatten()), label="AdaBoost")

ax.scatter(reshapeTreatment(y1),accuracyY1, reshapeTreatment(Y_test.flatten()), label="Decision Tree Classifier")

ax.scatter(reshapeTreatment(x2), accuracyX2, reshapeTreatment(Y_test.flatten()), label="AdaBoost 2")
plt.legend()
plt.show()

