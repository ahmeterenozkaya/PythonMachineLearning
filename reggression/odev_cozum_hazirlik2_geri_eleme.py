#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#1. kutuphaneler
from cgi import test
from matplotlib import axis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
from sklearn.preprocessing import OneHotEncoder
#veri önişleme
veriler = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python Machine Learning/odev_tenis.csv')
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

c = veriler2.iloc[:,:1]
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()

havadurumu = pd.DataFrame(data=c, index=range(14),columns=['o','r','s'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis=1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler],axis=1)

x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33,random_state=0)

reggessor = LinearRegression()
reggessor.fit(x_train,y_train)
y_pred = reggessor.predict(x_test)

X = np.append(arr = np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()

sonveriler = sonveriler.iloc[:,1:]


X = np.append(arr = np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)


X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

reggessor.fit(x_train,y_train)
y_pred = reggessor.predict(x_test)


# print(veriler)
# print(veriler2)
# print(c)
# print(havadurumu)
# print(sonveriler)
# print(x_train)
# print(x_test)
# print(y_train)
print(y_test)
print(y_pred)

# print(model.summary())