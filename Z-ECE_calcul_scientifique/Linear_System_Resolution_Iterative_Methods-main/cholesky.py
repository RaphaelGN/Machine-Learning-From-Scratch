#Algorithme de Cholesky
import numpy as np
def cholesky(A,b):
    n = len(A)
    L = np.zeros((n,n))
    L[0][0] = A[0][0]**0.5
    #on détermine la matrice L lower triangle tel que A = L*Lt
    for j in range(1,n):
        L[j][0] = A[0][j]/L[0][0]

    for i in range(1,n):

        sumSquare = 0
        for k in range(0,i):
            sumSquare += L[i][k]**2

        L[i][i] = (A[i][i] - sumSquare)**0.5

        for j in range(i+1,n):
            sumLL = 0
            for k in range(0,i):
                sumLL += L[i][k]*L[j][k]
            L[j][i] = (A[i][j] - sumLL)/L[i][i]
    print("L*Lt ",L.dot(np.transpose(L)))
    Lt = np.transpose(L)

    # Ax=b équivaut à LLtx=b d'où Ly=b avec y=Ltx et Lx=y
    x = np.zeros((n,1))
    y = np.zeros((n,1))
    y[0] = b[0]/L[0][0]
    for k in range(1,n):
        sumLY = 0
        for j in range(0,k):
            sumLY += L[k][j]*y[j]
        y[k] = (b[k] - sumLY)/L[k][k]

    for k in range(n-1,-1,-1):
        sumLtx = 0
        for j in range(k+1,n):
            sumLtx += Lt[k][j]*x[j]
        x[k] = (y[k] - sumLtx)/Lt[k][k]
    return x