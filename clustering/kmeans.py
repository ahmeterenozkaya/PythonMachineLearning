import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



veriler = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python-Machine-Learning/csvexamplefolder/musteriler.csv')

X = veriler.iloc[:,3:].values

kmeans = KMeans(n_clusters=3,init='k-means++') # for'u kullanıp plt.plot'da gösterdiğimizde x'e en yakın değeri n_clusters olarak alırız! # Unutma!
kmeans.fit(X)
sonuclar = []
for i in range(1,10):
    kmeans = KMeans (n_clusters =i, init='k-means++',random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    print(sonuclar)
    print('---------------------------------------')
    # print(i)

plt.plot(range(1,10),sonuclar)
plt.show()



# print(kmeans.cluster_centers_)
# print(veriler)