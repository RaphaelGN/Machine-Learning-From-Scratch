# résolution numérique d'une équation d'ordre 2 par la méthode des différences finies

#conditions aux bornes u(0)=1 u(1)=e

import numpy as np
from math import*
import cholesky
from matplotlib import pyplot as plt


def differences_finies(N,a,b,f):

    #pas de subdivision
    h=(b-a)/N

    # segment x
    x = np.linspace(a, b,N-1)

    # Discretisation
    xi = np.array([(a+(i*h)) for i in range(0,N-1)])

    #resoudre Au=b

    #construction matrice tridiagonale
    A = np.zeros((N-1,N-1))

    for i in range(N-1):
        for j in range(N-1):
            if(i==j):
                A[i][j]=2*(1+h**2)
            elif(j==i+1):
                A[i][j]=-1
            elif(i==j+1):
                A[i][j]=-1

    #construction vecteur_f
    vecteur_f = np.zeros((N-1,1))
    for i in range(N-1):
        if(i==0):
            vecteur_f[i] = f(xi[i])+1/h**2
        elif(i==N-2):
            vecteur_f[i] = f(xi[i])+exp(1)/h**2
        else:
            vecteur_f[i] = f(xi[i])

    #Résolution de l'équation linéaire à l'aide de Cholesky
    U = cholesky.cholesky((1/h**2)*A,vecteur_f)
    #print(U)
    U_exact=[ f(xi[i]) for i in range(0,N-1)]
    #print(U_exact)
    return U,U_exact,x,xi





# Definition de la fonction exponentielle (solution exacte)
def exponentielle(x):
    y= exp(x)
    return y


#nb de points
N=100

#Test avec la fonction sinus
U,U_exact,x,xi = differences_finies(N,0,1,exponentielle)

#graphique de la solution approchée VS la solution exacte
fig , ax = plt.subplots()
plt.plot(xi,U,label='Approximation Différence Finies')
plt.title("Approximation Différence Finies VS Solution exacte")
plt.plot(x,U_exact,label='Solution exacte')
leg = ax.legend();
plt.show()

#graphique de l'erreur entre la solution approchée et la solution exacte
fig , ax = plt.subplots()
erreur = abs(U-U_exact)
print(erreur)
plt.plot(xi,erreur,label='Erreur')
plt.title("Erreur Approximation Différence Finies VS solution exacte")
leg = ax.legend();
plt.show()


