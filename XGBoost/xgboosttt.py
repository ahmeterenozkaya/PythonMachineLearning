#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:03:40 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from xgboost import XGBClassifier

# veri kümesi
dataset = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python-Machine-Learning/csvexamplefolder/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

labelencoder = LabelEncoder()
X[:,1] = labelencoder.fit_transform(X[:,1])
labelencoder2 = LabelEncoder()
X[:,2] = labelencoder2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]


# eğitim ve test kümelerinin bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# XGBoost

classifier = XGBClassifier()
classifier.fit(X_train,y_train)


# Tahminler
y_pred = classifier.predict(X_test)

#  Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
