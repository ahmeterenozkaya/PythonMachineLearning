import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python Machine Learning/veriler.csv')
# print(veriler)


boy = veriler[['boy']]
# print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)