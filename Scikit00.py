#Establish Dataframe
import pandas as pd
#Plotting Data 
import matplotlib.pyplot as plt

#Number Py library to manipulate data
import numpy as np

#ADABOOST algorithm implementation
from sklearn.ensemble import AdaBoostClassifier
#ADABOOST Regressor - singular Training for Chemical molecule
from sklearn.ensemble import GradientBoostingClassifier
#Splits Data from Train/Test Data
from sklearn.model_selection import train_test_split

#Establish STUMP for Classifier WEIGHT recalibration 
from sklearn.tree import DecisionTreeClassifier
#Accuracy Score Metric for Classifier
from sklearn.metrics import accuracy_score



#ADAboost ---> 100 weak learners 
ada = AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=40,max_depth=24),n_estimators=100, algorithm='SAMME',
                                 learning_rate=.02)
#DecisionTreeClassifier -- MAX 40 leaf Nodes
dtc = DecisionTreeClassifier(max_leaf_nodes=40,max_depth=24)

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)

#csv File into allData Variable  ---> Later modify for TRAINING
allData = pd.read_csv("newNapchc13.csv")

#Input 'data' var for spliting to TRAINING 
data = allData

#Labels of the Treatment
dataY = data[['Treatment']]

#Data from Curticular Hydrocarbon
dataX = data.drop(["Treatment"], axis=1).values
#print(dataX,dataY)

chc23_1 = data[["c23:1"]]

def reshapeTreatment(x):
        
        lst = []
        for _ in range(len(x)):

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


dataY = np.array(dataY)
# Split dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.25) # 50% training and 50% test

ctest, ctrain = train_test_split(chc23_1, test_size=0.5)


#Training of Data/Prediction
adaboost = ada.fit(X_train,Y_train.flatten()).predict(X_test)
decTree = dtc.fit(X_train,Y_train.flatten()).predict(X_test)
gradBo = gbc.fit(X_train,Y_train.flatten()).predict(X_test)


accuracyadaboost = accuracy_score(Y_test.flatten(), adaboost)
accuracydecTree = accuracy_score(Y_test.flatten(), decTree)
accuracygradBoo = accuracy_score(Y_test.flatten(),gradBo)


#3D Figure 
ax = plt.axes(projection="3d")
ax.set_xlim(0,4)
ax.set_xlabel("Predicted Treatment")

ax.set_ylim(0,2)
ax.set_ylabel("Accuracy Score")

ax.set_zlabel("Actual Treatment")
ax.set_zlim(0,4)

#Converts String to INT for plotting 
adaboost = reshapeTreatment(adaboost)
decTree = reshapeTreatment(decTree)
Y_test = reshapeTreatment(Y_test.flatten())
gradBo = reshapeTreatment(gradBo)

ax.scatter(adaboost,accuracyadaboost,Y_test, label="AdaBoost")
ax.scatter(decTree,accuracydecTree, Y_test, label="Decision Tree Classifier")

ax.scatter(gradBo, accuracygradBoo,Y_test, label="Graduate Boosting Classifier")

plt.legend()
plt.show()
