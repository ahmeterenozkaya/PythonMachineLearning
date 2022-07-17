#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


veriler = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python Machine Learning/maaslar_yeni.csv')

x = veriler.iloc[:,2:3]
y = veriler.iloc[:,5:]
X = x.values
Y = y.values


#linear regression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)


#P-Value 

print('linear ols')
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())

#Linear Reggession
print('Linear R2 Değeri')
print(r2_score(Y,lin_reg.predict(x)))


#Polynomial Reggession
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

print('poly ols')
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())


print('Polynomial R2 Değeri')
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))


#SVR Reggession
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y)


svr_reg = SVR(kernel='rbf')
result = svr_reg.fit(x_olcekli,y_olcekli)

print('SVR ols')
model3 = sm.OLS(svr_reg.predict(x_olcekli,),x_olcekli)
print(model3.fit().summary())

print('SVR R2 Değeri')
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

# Decision Tree Regression
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
z = X + 0.5
k = X - 0.4

print('dt ols')
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())


print('DTR R2 Değeri')
print(r2_score(Y,r_dt.predict(X)))

#Random Forest Reggession
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0) # Estimators kaçtane karar ağacı olduğunu belirtir.
rf_reg.fit(X,Y.ravel())

print('rf ols')
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())


print('RFR R2 Değeri')
print(r2_score(Y,rf_reg.predict(X)))
print(r2_score(Y,rf_reg.predict(k)))
print(r2_score(Y,rf_reg.predict(z)))



# plt.show()
# print(veriler.cor())
# print(x,y)
# print(X,Y)