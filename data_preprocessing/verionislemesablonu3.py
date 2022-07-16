# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import statsmodels.api as sm


veriler = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python Machine Learning/veriler.csv')
ulke = veriler.iloc[:,0:1].values
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
age = veriler.iloc[:,1:4].values

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()


c = veriler.iloc[:,-1:].values

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

c = ohe.fit_transform(c).toarray()

country = pd.DataFrame(data=ulke, index= range(22), columns=['fr','tr','us'])
boykilo = pd.DataFrame(data=age,index=range(22),columns=['boy','kilo','yas'])
cinsiyet = veriler.iloc[:,-1].values

cinsiyett =pd.DataFrame(data=c[:,:1], index = range(22), columns=['cinsiyet'])

s=pd.concat([country,boykilo], axis=1)
s2=pd.concat([s,cinsiyett],axis = 1)

x_train, x_test, y_train, y_test=train_test_split(s,s2,test_size=0.33,random_state=0)

sc = StandardScaler()

result = X_train = sc.fit_transform(x_train) 
result1 = X_train = sc.fit_transform(x_test)

regressor = LinearRegression()
result = regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:,3:4].values
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]
veri = pd.concat([sol,sag],axis=1)

x_train, x_test, y_train, y_test=train_test_split(veri,boy,test_size=0.33,random_state=0)


l2 = LinearRegression()
l2.fit(x_train,y_train)
y_predd = l2.predict(x_test)

# X = np.append(arr= np.ones((22,1)).astype(int), values=veri, axis=1)
# X_l = veri.iloc[:,[0,1,2,3,4,5]].values
# X_l = np.array(X_l,dtype=float)
# model = sm.OLS(boy,X_l).fit()

X = np.append(arr= np.ones((22,1)).astype(int), values=veri, axis=1)
X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()



# print(x_train)
# print(y_train,y_predd)
print(veri)
print(X)
print(X_l)
print(model.summary())