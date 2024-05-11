#Algorithm to find the most probable sequence of hidden states z that would have generated a given sequence of observable states x

import numpy as np

#nb of hidden states
N=3

#nb of observable states
T=4

#given sequence: x = {x1,x3,x2,x0}
x = [1,3,2,0]

#Matrix aij (transition probability from z(t-1)=zi to z(t)=zj
aij = np.array([[1,0,0,0],
                [0.2,0.3,0.1,0.4],
                [0.2,0.5,0.2,0.1],
                [0.7,0.1,0.1,0.1]])
print("aij")
print(aij)

#Matrix bjk (emission probability from z(t)=zj to x(t)=xk
bjk = np.array([[1,0,0,0,0],
                [0,0.3,0.4,0.1,0.2],
                [0,0.1,0.1,0.7,0.1],
                [0,0.5,0.2,0.1,0.2]])
print("bjk")
print(bjk)

#initialization
alphaMatrix = np.zeros((N+1,T+1))

for j in range(0,N+1):
    if j == 1:
        alphaMatrix[j][0] = 1
    else:
        alphaMatrix[j][0] = 0

#HMM forward algorithm
def HMM(T,N,alphaMatrix,aij,bjk):
    path = []
    for t in range(1,T+1):
        for j in range(0,N+1):
            sum = 0
            for i in range(0,N+1):
                sum+= alphaMatrix[i][t-1]*aij[i][j]
            alphaMatrix[j][t] = bjk[j][x[t-1]]*sum
        jprim = np.argmax( alphaMatrix[:,t])
        path.append(alphaMatrix[jprim][t])
    print("path")
    return path

result = HMM(T,N,alphaMatrix,aij,bjk)
print(result)
