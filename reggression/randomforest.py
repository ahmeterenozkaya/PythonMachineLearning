import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

veriler = pd.read_csv('/home/kontrpars/Desktop/eren/Python/Python Machine Learning/maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
# plt.show()

#--------- Polynomial Regression -----------------

poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))


#---- SVR(Support Vector Reggression)

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y)


svr_reg = SVR(kernel='rbf')
result = svr_reg.fit(x_olcekli,y_olcekli)
plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')

# print(svr_reg.predict([[11]]))


#--------Desicion Tree-----------------

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
z = X + 0.5
k = X - 0.4
plt.scatter(X,Y,color='red')
plt.plot(X,r_dt.predict(x),color='blue')

plt.plot(X,r_dt.predict(z),color='green')
plt.plot(X,r_dt.predict(k),color='yellow')

# print(r_dt.predict([[11]]))
# print(r_dt.predict([[6.6]]))

# --------- Random Forest ---------

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0) # Estimators kaçtane karar ağacı olduğunu belirtir.
rf_reg.fit(X,Y.ravel())

plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(x),color='blue')

plt.plot(X,rf_reg.predict(z),color='green')
plt.plot(X,rf_reg.predict(k),color='green')


print(rf_reg.predict([[6.6]]))



# print(result)
plt.show() 