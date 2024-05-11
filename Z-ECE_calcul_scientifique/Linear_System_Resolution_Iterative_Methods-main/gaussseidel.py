#Méthode de Gauss-Seidel
#La méthode de Gauss-Seidel est une méthode itérative de résolution d'un système linéaire  de la forme Ax=b, ce qui signifie qu'elle génère une suite qui converge vers une solution de cette équation
#A = D(diagonale) - M(triangle inf) - N(triangle sup)
#Résolvons le système linéaire Ax=b
#ce qui revient à x(k+1) = (D-M)^-1*N*x(k)+(D-M)^-1*b

##Part1
import numpy as np
import math
import cholesky

def gaussseidel(A,b,Imax,errSeuil,x0):
    #i nb itérations effectuées
    iterNumber = 0
    #Trouver D,M,N tel que A = D-M-N
    D = np.diag(np.diag(A))
    M =A.copy()
    n=len(D)
    for i in range(0,n):
        for j in range(0,n):
            if j>=i:
                M[i,j]=0

    N=A.copy()
    n=len(D)
    for i in range(0,n):
        for j in range(0,n):
            if j<=i:
                N[i,j]=0
    print(D)
    print(M)
    print(N)
    print(D+M+N)

    #we compute x until convergence
    n = len(A)
    x = x0
    while iterNumber<Imax:
        xprec = x
        inverseMatrix = np.linalg.inv(D+M)
        DMN = inverseMatrix.dot(-N)
        DMNX = DMN.dot(x)
        DMB = inverseMatrix.dot(b)
        x =  DMNX + DMB
        err = np.linalg.norm(x-xprec)**2/np.linalg.norm(xprec)**2
        if( err < errSeuil):
            return x,iterNumber
        iterNumber += 1

    print("La méthode de Gauss-seidel n'a pas convergé'")



#A matrice carré inversible
A = np.array([[4,1,2],
              [1,6,3],
              [2,3,8]
    ])

#b vecteur
b = np.array([[10],
            [5],
            [4]])

#Imax nb max itérations
Imax= 100

#e précision d'arret de boucle
errSeuil = 0.00000001


#xo vecteur initial
x0 = np.array([[1],
            [1],
            [1]])



gaussseidel(A,b,Imax,errSeuil,x0)
x,iteration = gaussseidel(A,b,Imax,errSeuil,x0)
print("Gauss Seidel")
print("valeur obtenue de x")
print(x)
print("nb iteration")
print(iteration)

test = np.linalg.solve(A, b)
print("valeur réelle résolution linéaire")
print(test)


##Part 2

#A étant à diagonale dominante, d'après le théorème Gauss-Seidel va converger

n = 10

b = np.ones((n,1))

A = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        if(i==j):
            A[i][j]=3
        if(j==i+1):
            A[i][j]=1
        if(i==j+1):
            A[i][j]=1

#print("A ",A)
#print("b ",b)

#on vérifie que A est à diagonale dominante
def diagonaleDominante(A):
    for i in range(n):
        for j in range(n):
            sum =0
            for k in range(n):
                if(k!=i):
                    sum += abs(A[i][k])
        if(sum > abs(A[i][i])):
                return False
    return True

testDiagonaleDominante = diagonaleDominante(A)
print("A est elle à diagonale dominante?")
print(testDiagonaleDominante)


##Résolution de Ax=b avec la méthode Cholesky

print("Méthode de Cholesky")
x = cholesky.cholesky(A,b)
print("valeur obtenue de x")
print(x)

##Vérification résultat
test = np.linalg.solve(A, b)
print("valeur réelle résolution linéaire")
print(test)

##Résolution de Ax=b avec la méthode de Gauss-Seidel
print("Méthode de Gauss-Seidel")
x0 = np.ones((10,1))
x = gaussseidel(A,b,Imax,errSeuil,x0)
print("valeur obtenue")
print(x)



