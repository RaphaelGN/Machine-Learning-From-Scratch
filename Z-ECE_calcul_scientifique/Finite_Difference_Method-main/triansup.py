import numpy as np

def triansup(T,b):
# T est la matrice triangulaire inferieure et b le vecteur second membre.
#on résout Tx=b
    n=len(b);
    x = np.zeros((n,1))
    #on initialise le premier
    x[n-1]= b[n-1]/T[n-1][n-1]
    for i in range(n-2,-1,-1):
        sumTX = 0;
        for j in range (i+1,n):
            sumTX = sumTX + T[i][j]*x[j]
        x[i]= (1/T[i][i])*(b[i] -sumTX)
    return x;


T = np.array([[2,2,6],
              [0,3,5],
              [0,0,6]
    ])
b = np.array([[10],
            [2],
            [4]])

x = triansup(T,b)
print("triangularisation sup")
print("valeur obtenue")
print(x)

test = np.linalg.solve(T, b)
print("valeur réelle")
print(test)