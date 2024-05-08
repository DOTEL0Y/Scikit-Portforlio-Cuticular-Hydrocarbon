import sklearn.svm as svm
from sklearn.metrics import accuracy_score 
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib as plt
allData = pd.read_csv("napchc13.csv")
dataY = allData[['Treatment']].values
dataX = allData.drop(['Treatment','Queen'], axis=1).values

xTest, xTrain, yTest, yTrain = train_test_split(dataX, dataY.flatten(), train_size=0.5)
classifier = svm.LinearSVC(penalty='l2', dual='auto')
model = classifier.fit(xTrain,yTrain)
model = model.predict(xTest)

#View Presence of Hydrocarbon 21 in Queen Samples 
graphc21 = sns.catplot(
    data=allData, 
    kind='swarm', x='Treatment', y='Queen', hue='c21'

)
graphc21.savefig('c21.png')

