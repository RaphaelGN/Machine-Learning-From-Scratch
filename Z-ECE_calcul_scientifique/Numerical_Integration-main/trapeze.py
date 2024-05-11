# fonction qui à partir des points d’interpolation (xi) et des valeurs yi = f(xi), permet d’approximer intégrale I par la méthode des trapèzes

import numpy as np
from math import*
from scipy.integrate import quad

def trapeze(N,a,b,f):

    #pas de subdivision
    h=(b-a)/N

    # segment x
    x = np.linspace(a, b,N+1)

    # Discretisation
    xi = np.array([(a+(i*h)) for i in range(0,N)])

    #formule de Trapèze composéee:
    sumImages=0
    for i in range(0,N-1):
        sumImages += f(xi[i])+f(xi[i+1])

    I = h/2*sumImages

    return I


# Definition de la fonction sinus
def sinus(x):
    y= sin(x)
    return y

# Definition de la fonction exponentielle
def exponentielle(x):
    y= exp(-x**2)
    return y


#nb de points
N=1000

#Test avec la fonction sinus
Integrale = trapeze(N,0,pi,sinus)
print("Valeur intégrale approximée par la méthode des Trapèze pour sinus: ")
print(Integrale)
print("Valeur Réelle de l'intégrale: '")
res, err = quad(sin, 0,pi)
print(res)
print("erreur: ", abs(Integrale-res))

#Test avec la fonction exponentielle
Integrale = trapeze(N,0,7,exponentielle)
print("Valeur intégrale approximée par la méthode des Trapèze pour exponentielle: ")
print(Integrale)
print("Valeur Réelle de l'intégrale: '")
res, err = quad(exponentielle, 0,7)
print(res)
print("erreur: ", abs(Integrale-res))
#plus on augmente N, plus la valeur approximée se rapproche de la valeur réelle de l'integrale
