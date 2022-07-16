
#1.kutuphaneler
from mimetypes import init
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow import keras

veriler = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python-Machine-Learning/csvexamplefolder/Churn_Modelling.csv')

X = veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values

#0-1 dönüştürme

#Geography transform 0-1
le = preprocessing.LabelEncoder() 
X[:,1] = le.fit_transform(X[:,1])

#Gender transform 0-1(Male,Female)
le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

#Column transform numeric value

ohe = ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])],remainder='passthrough')
X = ohe.fit_transform(X)
X = X[:,1:]

#train_test
x_train, x_test,y_train, y_test= train_test_split(X,Y,test_size=0.33,random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# Keras

classifier = Sequential()
classifier.add(Dense(6,init = "uniform", activation='relu',input_dim = 11))
classifier.add(Dense(6,init = "uniform", activation='relu'))
classifier.add(Dense(1,init = "uniform",activation='sigmoid'))


classifier.compile(optimezer='adam')



# print(X_train)
# print(X_test)
# print(veriler)






