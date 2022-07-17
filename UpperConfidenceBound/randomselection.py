import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

veriler = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python-Machine-Learning/csvexamplefolder/Ads_CTR_Optimisation.csv')

N = 10000
d = 10
toplam = 0
secilenler = []
for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad] # Verilerdeki n. satır =1 ise ödül 1
    toplam = toplam + odul

    print(toplam)

plt.hist(secilenler)
plt.show()




# print(veriler)