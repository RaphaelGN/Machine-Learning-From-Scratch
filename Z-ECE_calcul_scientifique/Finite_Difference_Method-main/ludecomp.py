import numpy as np
from scipy.linalg import lu

def ludecomp(A):
# A est la matrice carré que l'on exprimer sous la forme LU tel que A=LU (L= matrice triangulaire inf et U= matrice triangulaire sup)
    n=len(A);
    L = np.zeros((n,n))
    U = np.zeros((n,n))
 # Decomposing matrix into Upper
    # and Lower triangular matrix
    for i in range(n):
        # Upper Triangular
        for k in range(i, n):

            # Summation of L(i, j) * U(j, k)
            sum = 0;
            for j in range(i):
                sum += (L[i][j] * U[j][k]);

            # Evaluating U(i, k)
            U[i][k] = A[i][k] - sum;

        # Lower Triangular
        for k in range(i, n):
            if (i == k):
                L[i][i] = 1; # Diagonal as 1
            else:

                # Summation of L(k, j) * U(j, i)
                sum = 0;
                for j in range(i):
                    sum += (L[k][j] * U[j][i]);

                # Evaluating L(k, i)
                L[k][i] = (A[k][i] - sum) / U[i][i];
    return L,U;


A = np.array([[2,-1,-2],
              [-4,6,3],
              [-4,-2,8]
    ])


L,U = ludecomp(A)
print("algorithme LU: ")
print("valeur obtenue L")
print(L)
print("valeur obtenue U")
print(U)


