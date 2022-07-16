# -*- coding: utf-8 -*-

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


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

cm = confusion_matrix(y_test,y_pred)

knn = KNeighborsClassifier(n_neighbors=1,metric='minkowski') # n_neighbors komşu sayısına bakar. 
knn.fit(X_train,y_train)

knn_y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,knn_y_pred)

from sklearn.svm import SVC

# svc = SVC(kernel='linear')
svc = SVC(kernel='rbf')

svc.fit(X_train,y_train)

svc_y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,svc_y_pred)


gnb = GaussianNB()
gnb.fit(X_train,y_train)

gnb_y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test,gnb_y_pred)


dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train,y_train)
dtc_ypred = dtc.predict(X_test)
cm = confusion_matrix(y_test,dtc_ypred)


print('DTC')
print(cm)
print(dtc_ypred)
# print(x,y)
# print('----------------X_train--------------------------')
# print(X_train)
# print('----------------X_test--------------------------')
# print(X_test)
# print('----------------x_test--------------------------')
# print(x_test)
# print('----------------y_pred--------------------------')
# print(y_pred)
# print('----------------y_test--------------------------')
# print(y_test)






