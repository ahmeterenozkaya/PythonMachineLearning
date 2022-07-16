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



veriler = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python Machine Learning/satislar.csv')
# # print(veriler)

aylar = veriler[['Aylar']]
# # print(aylar)

satislar = veriler[['Satislar']]
# # print(satislar)

result = x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

sc=StandardScaler()

        

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.transform(y_test)


lr = LinearRegression()
lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")


plt.show()


# print(tahmin)
# print(lr)
# y_pred = logr.predict(X_test)
# print(y_pred)
# print(y_test)
# print(veriler)


















