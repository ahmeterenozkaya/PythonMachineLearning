# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression



veriler = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python Machine Learning/veriler.csv')

x = veriler.iloc[:,1:4].values # bağımsız değişkenler
y = veriler.iloc[:,4:].values # bağımlı değişkenler


# %33 test, %67 train için ayrılacak.. 
result = x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

sc=StandardScaler()

        
#fit eğitme, transform eğitimi kullanma(eğitimi dönüştürme)
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)


print(x,y)
print('----------------X_train--------------------------')
print(X_train)
print('----------------X_test--------------------------')
print(X_test)
print('----------------x_test--------------------------')
print(x_test)
print('----------------y_pred--------------------------')
print(y_pred)
print('----------------y_test--------------------------')
print(y_test)






