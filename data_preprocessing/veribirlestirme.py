import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

veriler = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python Machine Learning/veriler.csv')
# imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
age = veriler.iloc[:,1:4].values
# imputer = imputer.transform(age[:,1:4])

le = preprocessing.LabelEncoder()
ulke = veriler.iloc[:,0:1].values

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

# print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

country = pd.DataFrame(data=ulke, index= range(22), columns=['fr','tr','us'])
boykilo = pd.DataFrame(data=age,index=range(22),columns=['boy','kilo','yas'])
cinsiyet = veriler.iloc[:,-1].values

cinsiyett =pd.DataFrame(data=cinsiyet, index = range(22), columns=['cinsiyet'])

s=pd.concat([country,boykilo], axis=1)
s2=pd.concat([s,cinsiyett],axis = 1)

print(s2)