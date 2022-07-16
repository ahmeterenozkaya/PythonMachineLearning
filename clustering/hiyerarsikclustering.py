import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

veriler = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python-Machine-Learning/csvexamplefolder/musteriler.csv')

X = veriler.iloc[:,3:].values

kmeans = KMeans(n_clusters=3,init='k-means++') # for'u kullanıp plt.plot'da gösterdiğimizde x'e en yakın değeri n_clusters olarak alırız! # Unutma!
kmeans.fit(X)
sonuclar = []
for i in range(1,10):
    kmeans = KMeans (n_clusters =i, init='k-means++',random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    # print(sonuclar)
    # print('---------------------------------------')

# plt.plot(range(1,10),sonuclar)
# plt.show()

kmeans = KMeans (n_clusters =4, init='k-means++',random_state=123)
y_tahmin = kmeans.fit_predict(X)
plt.scatter(X[y_tahmin==0,0],X[y_tahmin==0,1],s=100,c='red')
plt.scatter(X[y_tahmin==1,0],X[y_tahmin==1,1],s=100,c='blue')
plt.scatter(X[y_tahmin==2,0],X[y_tahmin==2,1],s=100,c='green')
plt.scatter(X[y_tahmin==3,0],X[y_tahmin==3,1],s=100,c='yellow')
# plt.title('Kmeans')
# plt.show()



# ------------------------------------------------------------



# Hiyerarşik Clustering
ac = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
ac_ypredict = ac.fit_predict(X)

plt.scatter(X[ac_ypredict==0,0],X[ac_ypredict==0,1],s=100,c='red')
plt.scatter(X[ac_ypredict==1,0],X[ac_ypredict==1,1],s=100,c='blue')
plt.scatter(X[ac_ypredict==2,0],X[ac_ypredict==2,1],s=100,c='green')
plt.scatter(X[ac_ypredict==3,0],X[ac_ypredict==3,1],s=100,c='yellow')
plt.title('HC')

# plt.show()
# print(ac_ypredict)



#----------------------------------------------------------------------
# Scipy Cluster Hierarchy (dendogram)
x = sch.dendrogram(sch.linkage(X,method='ward'))

plt.show()

