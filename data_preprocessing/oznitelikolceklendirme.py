import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

veriler = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python Machine Learning/veriler.csv')
age = veriler.iloc[:,1:4].values
le = preprocessing.LabelEncoder()
ulke = veriler.iloc[:,0:1].values

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

country = pd.DataFrame(data=ulke, index= range(22), columns=['fr','tr','us'])
boykilo = pd.DataFrame(data=age,index=range(22),columns=['boy','kilo','yas'])
cinsiyet = veriler.iloc[:,-1].values

cinsiyett =pd.DataFrame(data=cinsiyet, index = range(22), columns=['cinsiyet'])

s=pd.concat([country,boykilo], axis=1)
s2=pd.concat([s,cinsiyett],axis = 1)

x_train, x_test, y_train, y_test=train_test_split(s,s2,test_size=0.33,random_state=0)

sc = StandardScaler()

result = X_train = sc.fit_transform(x_train) 
result1 = X_train = sc.fit_transform(x_test)

print(result)
print(result1)