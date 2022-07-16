from distutils.command.config import config
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


veriler = pd.read_excel('Iris.xls')

x = veriler.iloc[:,:1:4].values
y = veriler.iloc[:,4:].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
logr_ypred = logr.predict(X_test)
logr_cm = confusion_matrix(y_test,logr_ypred)

# print('LR')
# print(logr_ypred)
# print(logr_cm)
# print(y_test)


# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors =1,metric ='minkowski')
knn.fit(X_train,y_train)
knn_ypred = knn.predict(X_test)
knn_cm = confusion_matrix(y_test,knn_ypred)

# print('KNN')
# print(knn_ypred)
# print(knn_cm)

# SVC(Support Vector Classification)
from sklearn.svm import SVC 

svc = SVC(kernel='poly')
svc.fit(X_train,y_train)
svc_ypred = svc.predict(X_test)
svc_cm = confusion_matrix(y_test,svc_ypred)

# print(svc_ypred)
# print('SVC')
# print(svc_cm)

#Naive Bayes 
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train,y_train)
gnb_ypred = gnb.predict(X_test)
gnb_cm = confusion_matrix(y_test, gnb_ypred)

# print(gnb_ypred)
# print('GNB')
# print(gnb_cm)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
dtc_ypred = dtc.predict(X_test)
dtc_cm = confusion_matrix(y_test,dtc_ypred)

# print(dtc_ypred)
# print('DTC')
# print(dtc_cm)

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10,criterion ='entropy')
rfc.fit(X_train,y_train)
rfc_ypred=rfc.predict(X_test)
rfc_cm = confusion_matrix(y_test,rfc_ypred)

# print(rfc_ypred)
# print('RFC')
# print(rfc_cm)

# ROC, TPR(True Positive Rating), FPR(False Positive Rating)
y_proba = rfc.predict_proba(X_test)

# print(y_test)
# print(y_proba[:,0])

#Metrics
from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='None')

print(fpr)
print(tpr)








# print(x,y)
# print(veriler)
