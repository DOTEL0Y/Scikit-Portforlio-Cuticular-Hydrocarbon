#Written by Oscary Antonio Dotel
#Script to augment napchc13 csv file for greater Leave-One-Out product
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

origData = pd.read_csv('napchc13 copy.csv')
newData = origData.drop(['Queen'],axis=1)
# print(origData.info())

# newData = origData.drop(['Queen'], axis=1)

#Splices of Treatment from Original Data - STILL DATAFRAME
c = origData.iloc[:9,2:26]
dp = origData.iloc[10:16,2:26]
im = origData.iloc[17:26,2:26]
imdp = origData.iloc[27:33,2:26]
pyr = origData.iloc[34:40,2:26]
cc21 = c['c23'].values
c21 = origData['c21'].values
jcolumn = normalize(cc21.reshape(1,-1))
# print(normalize(c21.reshape(1,-1)))

def augmentTreatmentData(treatmentData,treatment):
    row = [treatment]
    ester34 = newData['esters34'].values
    ester34 = normalize(ester34.reshape(1,-1))
    for column  in range(25):
        if column > 0:
             j = newData.columns[column]
             chem = treatmentData[j].values
             chem = normalize(chem.reshape(1,-1))
             p50 = np.percentile(chem,50)
             lowR  = p50 - chem.mean()
             highR = p50 + chem.mean()  
             if lowR - chem.mean() < 0:
                 lowR = 0  
             row.append(np.random.uniform(low = lowR, high = highR))
             #print(row)
    p50 = np.percentile(ester34,50)
    lowR = p50 - ester34.mean()
    highR = p50 + ester34.mean()
    if lowR - ester34.mean() < 0:
        lowR = 0
    row.append(np.random.uniform(low=lowR,high=highR))
    newData.loc[len(newData.index)] = row


for i in range(20):
    augmentTreatmentData(c,'C')
    augmentTreatmentData(dp,'DP')
    augmentTreatmentData(im,'IM')
    augmentTreatmentData(imdp,'IMDP')
    augmentTreatmentData(pyr,'PRY')

newData.to_csv('1newNapchc13.csv')
#27 Columns / 26 Excluding Queen Column --- Function to normalize data from rows based on Treatment and normalize.
#Then, construct 200 new samples between (Median + Range, Median - Range) -- Insert to into New Dataframe.
 
# standard = normalize(origData.drop(['Treatment','Queen'],axis=1).values)
# newDataframe = pd.DataFrame(standard, columns= index)

# for n in range(1,100):
#     standard.append(np.random.uniform(size = (1,25),low=.01,high=.1))