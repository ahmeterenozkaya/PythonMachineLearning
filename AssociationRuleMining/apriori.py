# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import jaccard_score
from apyori import apriori

veriler = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python-Machine-Learning/csvexamplefolder/sepet.csv',header=None)

t=[]
for i in range (0,7501):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])


result = apriori(t,min_support=0.01,min_confidence=0.2,min_lift=3,min_length=2)
print(list(result))