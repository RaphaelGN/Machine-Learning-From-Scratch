#on veut résoudre Ax=b pour toutes matrices A via la méthode de Gauss
import numpy as np
import triansup


def gauss(A,b):
    n = len(b)
    M = np.zeros((n,n))
    #triangularisation de la matrice A et b
    for k in range(0,n):
        for i in range (k+1,n):
            M[i][k] = A[i][k]/A[k][k]
            for j in range (k,n):
                A[i][j] = A[i][j] - M[i][k]*A[k][j]
            b[i] = b[i] - M[i][k]*b[k]
    #appel résolution système linéaire avec matrice supérieure
    x = triansup.triansup(A,b)
    return x;



A = np.array([[8,1,7],
              [2,2,4],
              [6,4,6]
    ])
b = np.array([[10],
            [5],
            [4]])

x = gauss(A,b)
print("Gauss")
print("valeur obtenue")
print(x)

test = np.linalg.solve(A, b)
print("valeur réelle")
print(test)

print("A/b")
print(A/b)

print("inv(A)*b")
print(np.invert(A)*b)