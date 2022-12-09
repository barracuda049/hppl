import imp
from this import d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder

cols = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']

df = pd.read_csv("./magic04.data", names= cols) #read csv 
df['class'] = (df['class'] == 'g').astype(int) # assign 1 to all g, 0 to h
print(df.head())

# for label in cols[:-1]:
#     plt.hist(df[df['class']==1][label], color='blue', label='gamma', alpha=0.7, density=True)
#     plt.hist(df[df['class']==0][label], color='red', label='h', alpha=0.7, density=True)
#     plt.title(label)
#     plt.ylabel('Probability')
#     plt.xlabel(label)
#     plt.legend()
#     plt.show()

# To create datasets train, validation, test datasets 
#DataFrame. sample - Возврат случайной выборки элементов с оси объекта

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))]) # split mean to split the df to [:int(0.6*len(df))], 
#[int(0.6*len(df)):int(0.8*len(df))], [int(0.8*len(df)):]

# print("len(df) - ", len(df), "len(train) - ", len(train), "len(valid) - ", len(valid), "len(test) - ", len(test))

def scale_dataframe(dataframe, oversample=False): #some of the values are small , and as I understand to scale these values 
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    ros = RandomOverSampler()
    X,y = ros.fit_resample(X,y)

    data = np.hstack((X, np.reshape(y, (-1, 1)))) # воссоединение исходниго набора данных, (-1,1) == (2,1)
    return data, X, y

train, X_train, y_train = scale_dataframe(train, oversample=True)
valid, X_valid, y_valid = scale_dataframe(valid, oversample=False)
test, X_test, y_test = scale_dataframe(test, oversample=False)

# f = np.array([[1,2],[1,2]])
# g = np.array([1,2])

# print(np.reshape(g, (2,1)))

#KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

print(classification_report(y_test, y_pred))