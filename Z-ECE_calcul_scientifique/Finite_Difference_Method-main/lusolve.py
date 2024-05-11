#on veut résoudre Ax=b pour toutes matrices A vi la méthode LU
import numpy as np
import triansup
import trianinf
import ludecomp


def lusolve(A,b):
    L,U = ludecomp.ludecomp(A)
    y = trianinf.trianinf(L,b)
    x = triansup.triansup(U,y)
    return x;



A = np.array([[8,1,7],
              [2,2,4],
              [6,4,6]
    ])
b = np.array([[10],
            [5],
            [4]])

x = lusolve(A,b)
print("Algorithme LU")
print("valeur obtenue")
print(x)

test = np.linalg.solve(A, b)
print("valeur réelle")
print(test)

print("A/b")
print(A/b)

print("inv(A)*b")
print(np.invert(A)*b)