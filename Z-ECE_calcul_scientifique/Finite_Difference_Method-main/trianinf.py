import numpy as np

def trianinf(T,b):
# T est la matrice triangulaire inferieure et b le vecteur second membre.
#on résout Tx=b
    n=len(b);
    print(n);
    print(T.shape)
    x = np.zeros((n,1))
    #on initialise le premier
    x[0]= b[0]/T[0][0]
    #attention n exclusif avec la fonction range donc jusqu'à n-1
    for i in range(1,n):
        sumTX = 0;
        #attention i exclusif avec la fonction range donc jusqu'à i-1
        for j in range (0,i):
            sumTX = sumTX + T[i][j]*x[j]
        x[i]= (1/T[i][i])*(b[i] -sumTX)
    return x;


T = np.array([[2,0,0],
              [2,2,0],
              [6,4,6]
    ])
b = np.array([[10],
            [2],
            [4]])

x = trianinf(T,b)
print("triangularisation inf")
print("valeur obtenue")
print(x)

test = np.linalg.solve(T, b)
print("valeur réelle")
print(test)