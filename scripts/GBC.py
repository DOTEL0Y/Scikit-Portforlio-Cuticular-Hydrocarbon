import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.ensemble import AdaBoostClassifier as ada
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


allData = pd.read_csv("newNapchc13.csv")

cp = allData[:16]
cpTreatment = cp[["Treatment"]]
cp = cp.iloc[0:len(cp),2:26]


im_dp = allData[17:33]
im_dpTreatment = im_dp[["Treatment"]]
im_dp = im_dp.iloc[0:len(im_dp),2:26]

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
            else:
                lst.append([0,0,1]) 
        print(lst)
        return lst
        



cp_data_Test, cp_data_Train,cpt_TreTest, cpt_TreTrain  = train_test_split(cp, cpTreatment,train_size=.5)

i_data0, i_data1, it_data0, it_data1 = train_test_split(im_dp, im_dpTreatment, train_size=0.5)

cptTrain = cpt_TreTrain.to_numpy()
cptTest = cpt_TreTest.to_numpy()

Gradient = gbc(n_estimators= 100, learning_rate=.05).fit(cp_data_Train,cptTrain.flatten()).predict(cp_data_Test)

Ada = ada(dtc(max_depth=40,max_leaf_nodes=80), n_estimators=100, learning_rate=.1, algorithm="SAMME").fit(cp_data_Train,cptTrain.flatten()).predict(cp_data_Test)
print(cptTest.flatten())
print(Gradient)
print(Ada)
Ada=reshapeTreatment(Ada)

print(gbc(n_estimators= 100, learning_rate=.05).fit(cp_data_Train,cptTrain.flatten()).predict_proba(cp_data_Test))
print(ada(dtc(max_depth=40,max_leaf_nodes=80), n_estimators=100, learning_rate=.1, algorithm="SAMME").fit(cp_data_Train,cptTrain.flatten()).predict_proba(cp_data_Test))
gbcA = accuracy_score(cptTest.flatten(),Gradient)*100
adaA = accuracy_score(cptTest.flatten(),Ada)*100

print(gbcA, adaA)


# ddd =plt.axes(projection="3d")

# plt.figure(1)
# ddd.plot(reshapeTreatment(Ada),reshapeTreatment(cptTest.flatten()),adaA)
# ddd.scatter(reshapeTreatment(Ada),reshapeTreatment(cptTest.flatten()),adaA, color='r')
# print(len(gbc(n_estimators= 100, learning_rate=.05).fit(cp_data_Train,cptTrain.flatten()).predict_proba(cp_data_Test)))
# print(len(ada(dtc(max_depth=40,max_leaf_nodes=80), n_estimators=100, learning_rate=.1, algorithm="SAMME").fit(cp_data_Train,cptTrain.flatten()).predict_proba(cp_data_Test)))
# ddd.plot(reshapeTreatment(Gradient),reshapeTreatment(cptTest.flatten()),gbcA)
# ddd.scatter(reshapeTreatment(Gradient),reshapeTreatment(cptTest.flatten()),gbcA, color='g')

# plt.show()


