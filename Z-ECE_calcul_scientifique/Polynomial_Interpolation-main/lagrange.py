#Lagrange
#calculons le polynome d’interpolation de Lagrange d’une fonction f definit sur un intervalle [a, b].
import numpy as np
import math
import matplotlib.pyplot as plt

def lagrange(x,x_values,y_values):
    n = len(x_values)
    #P polynôme de lagrange
    P = 0
    for i in range(0,n):
        #L le produit
        L = 1
        YL = 0
        for k in range(0,n):
            if k!=i:
                L = L*((x-x_values[k])/(x_values[i]-x_values[k]))
        YL = y_values[i]*L
        P += YL
    return P

#Définir un nombre de points N
N = 10

#définir une discretion
x = np.zeros((N,1))
for i in range(-N,N):
    x[i] = i/N

##Trouver les points interpolés pour sin(x)
y = np.sin(x)


x_test = np.linspace(0,1)
y_test = np.zeros((len(x_test),1))
for i in range(len(x_test)):
    y_test[i] = lagrange(x_test[i],x,y)


#Plot P and the function
plt.plot(x,y)
plt.plot(x_test, y_test)

plt.show()

##Trouver les points interpolés pour 1/1=x^2

y = 1/(1+x**2)


x_test = np.linspace(0,1)
y_test = np.zeros((len(x_test),1))
for i in range(len(x_test)):
    y_test[i] = lagrange(x_test[i],x,y)


#Plot P and the function
plt.plot(x,y)
plt.plot(x_test, y_test)

plt.show()

##Points de Tchebycheff
#définir une discretion
x = np.zeros((N,1))
for i in range(N):
    x[i] = math.cos(((2*i+1)*math.pi)/2*N)

y = 1/(1+x**2)


x_test = np.linspace(0,1)
y_test = np.zeros((len(x_test),1))
for i in range(len(x_test)):
    y_test[i] = lagrange(x_test[i],x,y)


#Plot P and the function
plt.plot(x,y)
plt.plot(x_test, y_test)

plt.show()



