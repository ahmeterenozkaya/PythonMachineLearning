import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

eksikveri = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python Machine Learning/eksikveriler.csv')
ortalama = eksikveri[['yas']].mean()

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

age = eksikveri.iloc[:,1:4].values
print(age)
imputers = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])
print(age)


