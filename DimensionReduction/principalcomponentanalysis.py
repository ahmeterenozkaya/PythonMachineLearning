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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python-Machine-Learning/csvexamplefolder/Wine.csv')
#pd.read_csv("veriler.csv")
# test
# print(veriler)

#veri on isleme

X= veriler.iloc[:,0:13].values
Y = veriler.iloc[:,13].values

#verilerin egitim ve test icin bolunmesi

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

#verilerin olceklenmesi

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# PCA

pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# PCA dönüşümünden önce gelen LR

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# PCA dönüşümünden sonra gelen LR

classifer2 = LogisticRegression(random_state=0)
classifer2.fit(X_train2,y_train)

#tahminler

y_pred = classifier.predict(X_test)
y_pred2 = classifer2.predict(X_test2)

# Actual PCA olmadan çıkan sonuç 
print("gerçek / PCA'siz")
cm = confusion_matrix(y_test,y_pred)
print(cm)

# Actual PCA sonrası çıkan sonuç 
print("gerçek / PCA ile")
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)

# PCA sonrası / PCA öncesi
print("PCA'siz / PCA'li")
cm3 = confusion_matrix(y_pred,y_pred2)
print(cm3)