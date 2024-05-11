# résolution numérique d'une équation d'ordre 2 par la méthode des différences finies

#conditions aux bornes u(0)=0 u(1)=0

import numpy as np
from math import*
import cholesky
import ludecomp
import lusolve
import triansup
import trianinf
import time


def differences_finies(N,a,b,f):

    #pas de subdivision
    h=(b-a)/N

    # segment x
    x = np.linspace(a, b,N+1)

    # Discretisation
    xi = np.array([(a+(i*h)) for i in range(0,N)])

    #resoudre Au=b

    #construction matrice tridiagonale
    A = np.zeros((N-1,N-1))

    for i in range(N-1):
        for j in range(N-1):
            if(i==j):
                A[i][j]=2
            if(j==i+1):
                A[i][j]=-1
            if(i==j+1):
                A[i][j]=-1

    #construction vecteur_f
    vecteur_f = np.zeros((N-1,1))
    for i in range(N-1):
        vecteur_f[i] = f(xi[i])

    #Résolution de l'équation linéaire à l'aide de Cholesky
    t = time.time()
    U = cholesky.cholesky((1/h**2)*A,vecteur_f)
    print(U)
    elapsed = time.time() - t
    print("temps écoulé Cholesky ",elapsed)


    #Résolution avec la factorisation LU
    t = time.time()
    U = lusolve.lusolve((1/h**2)*A,vecteur_f)
    print(U)
    elapsed = time.time() - t
    print("temps écoulé méthode LU ",elapsed)





# Definition de la fonction sinus
def sinus(x):
    y= sin(pi*x)
    return y


#nb de points
N=20

#Test avec la fonction sinus
differences_finies(N,0,1,sinus)

