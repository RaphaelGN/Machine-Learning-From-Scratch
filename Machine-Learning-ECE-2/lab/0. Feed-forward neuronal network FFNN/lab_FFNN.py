# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

data= np.loadtxt("data_ffnn_3classes.txt")

#print(type(float(data[1,0]))) 
L=[]
#print(len(data))
X1vec = []
X2vec = []
Yvec = []
for k in range(len(data)):
    X1vec.append(float(data[k,0])) 
    X2vec.append(float(data[k,1]))
    Yvec.append(float(data[k,2])) 
#print(X1vec)
#print(X2vec)
#print(Yvec)

#List of X1 and X2 where y=0
x10 =[]
x20 = []
#List of X1 and X2 where y=1
x11 =[]
x21 = []
#List of X1 and X2 where y=2
x12 =[]
x22 = []
for i in range(len(Yvec)):
    if(Yvec[i]==0):
        x10.append(X1vec[i])
        x20.append(X2vec[i])
    elif(Yvec[i]==1):
        x11.append(X1vec[i])
        x21.append(X2vec[i])
    else:
        x12.append(X1vec[i])
        x22.append(X2vec[i])
plt.plot(x10, x20, "ro", label = "y=0")
plt.plot(x11, x21,  "bo", label = "y=1")
plt.plot(x12, x22,  "go", label = "y=2")
plt.axis([0, 10, 0, 10])
plt.grid(True)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Actual output variable")
plt.legend()
plt.show()


#Matrix (1,71)
X1= np.array([X1vec])
X2= np.array([X2vec])
#Y= np.array([Y])

#Matrix (71,2)
X = np.concatenate((X1.T,X2.T),axis=1)
#print("X shape : ",X.shape)
O,N = X.shape
#print("Matrix X : \n",X)

I = np.ones((O,1))
#print("I shape : ",I.shape)
#print("Matrix I : \n",I)



Xb = np.concatenate((I,X),axis=1)

#print("Matrix Xbb : \n",Xb)
#print("Xb shape : ",Xb.shape)

#print(Xb[0][1])


SSE = []
min_SSE = 100
K= 10
#V = np.random.rand(3,K)
"""
V0 = np.random.uniform(-1,1,3*K)
V = np.reshape(V0, (3, K))
"""
V = np.array([[-0.50131844,  0.65838527, -0.60157993,  0.65302492,  0.294982  ,
        -0.84132428, -0.04401377, -0.48535996, -0.48050531,  0.12579815],
       [ 0.51153216,  0.53274982,  0.39745388, -0.97512427, -0.39862787,
        -0.17849767,  0.12963851,  0.98837412, -0.56181012, -0.69005636],
       [-0.6424869 , -0.92725727,  0.85884549, -0.1419165 ,  0.02142941,
        -0.41971407,  0.37333187, -0.20431422,  0.86330821,  0.3062777 ]])

v = np.copy(V)
#print("Matrix V : \n",V)
#print("V shape : ",V.shape)
J= 3 #And we have a classification problem
# because there are three kind of discret outputs possible {0,1,2} 

#W = np.random.rand(K+1,J)
"""
W0 = np.random.uniform(-1,1,(K+1)*J)
W = np.reshape(W0, (K+1, J))
"""
W = np.array([[ 0.98940933, -0.34778822,  0.18073294],
       [-0.30859373, -0.75752076,  0.10272493],
       [-0.5390539 ,  0.83161405,  0.00342125],
       [-0.36501278, -0.53333388, -0.29562208],
       [ 0.34750455, -0.45474235,  0.70110718],
       [-0.67197557, -0.48580859, -0.44963163],
       [-0.88702443, -0.10803092, -0.55720584],
       [-0.71052302, -0.16704237, -0.08458558],
       [-0.82211589, -0.07491992, -0.21067935],
       [ 0.22310839, -0.23244688,  0.22887257],
       [ 0.37120256,  0.03351009, -0.69234653]])

w = np.copy(W)
#print("Matrix W : \n",W)
#print("W shape : ",W.shape)
    

#------------------------------------------------------------------------------
#FORWARD PROPAGATION PART
iteration = 0
for iteration in range(150):
    iteration +=1
    #print("iteration : ",iteration )
    
    
    Xbb = np.matmul(Xb, V)
    #print("Matrix Xbb : \n",Xbb)
    #print("Xbb shape : ",Xbb.shape)
    
    
    F = 1/(1+np.exp(-Xbb))
    #print("Matrix F : \n",F)
    #print("F shape : ",F.shape)
    
    
    Fb = np.concatenate((I,F),axis=1)
    
     
    
    Fbb = np.matmul(Fb, W)
    #print("Matrix Fbb : \n",Fbb)
    #print("Fbb shape : ",Fbb.shape)
    
    G = 1/(1+np.exp(-Fbb))
    #print("Matrix G : \n",G)
    #print("G shape : ",G.shape)
    
    
    #y is a matrix (71,3)
    y = np.zeros((O,J)) 
    for i in range(O):
        if (Yvec[i]==0):
            y[i][0]=1 ##[1,0,0]
        elif (Yvec[i]==1):
            y[i][1]=1 ##[0,1,0]
        elif (Yvec[i]==2):
            y[i][2]=1##[0,0,1]
                
    
    #Sum of square errors
    E = 0
    for i in range(O):
        for j in range(J):
            E+=0.5*(G[i][j] -y[i][j])**2
    SSE.append(E)
    
    #print(SSE())
    
    
    
    #------------------------------------------------------------------------------
    ##BACK PROPAGATION PART
    
    alpha1= 0.1
    alpha2= 0.01
    for k in range(K+1):
        for j in range(J):
            Sum = 0
            for i in range(O):
                Sum+=(G[i][j] - y[i][j])*G[i][j]*(1-G[i][j])*Fb[i][k]
            W[k][j] = W[k][j] - alpha1*Sum
    
    for n in range(N+1):
        for k in range(K):
            Sum = 0
            for i in range(O):
                for j in range(J):
                    Sum += (G[i][j] - y[i][j])*G[i][j]*(1-G[i][j])*W[k][j]*Fb[i][k]*(1-Fb[i][k])*Xb[i][n]
            V[n][k] = V[n][k]-alpha2*Sum


    if E<min_SSE:
        min_SSE = E
        best_W = np.copy(W)
        best_V = np.copy(V)
        E = min_SSE



it = [i for i in range(1,len(SSE)+1)]
plt.plot(it,SSE)
plt.xlabel("Nb of iterations")
plt.ylabel("SSE")
plt.grid(True)
plt.show()

Ycheck = []
for j in range(0,O):
    max_val = 0
    """
    if G[j][1] > G[j][0]:
        max_val=1
    if G[j][2] > G[j][1]:
        max_val=2
    """
    max_val = max(G[j][0],G[j][1],G[j][2])
    if G[j][0]==max_val:
        max_val=0
    elif G[j][1]==max_val:
        max_val=1
    else:
        max_val=2
    Ycheck.append(max_val)
    
    
    
#List of X1 and X2 where y=0
x10 =[]
x20 = []
#List of X1 and X2 where y=1
x11 =[]
x21 = []
#List of X1 and X2 where y=2
x12 =[]
x22 = []
for i in range(len(Ycheck)):
    if(Ycheck[i]==0):
        x10.append(X1vec[i])
        x20.append(X2vec[i])
    elif(Ycheck[i]==1):
        x11.append(X1vec[i])
        x21.append(X2vec[i])
    else:
        x12.append(X1vec[i])
        x22.append(X2vec[i])
plt.plot(x10, x20, "ro", label = "y=0")
plt.plot(x11, x21,  "bo", label = "y=1")
plt.plot(x12, x22,  "go", label = "y=2")
plt.axis([0, 10, 0, 10])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Predicted output")
plt.legend()
plt.show()
    

        
print("Smallest SSE:", min_SSE)
print("Best W :\n", best_W)
print("Best V :\n", best_V)








#------------------------------------------------------------------------------
#NEW INPUT VARIABLE


X = np.array([[2,2],[4,4],[4.5,1.5]])
print("X :\n", X)
O,N = X.shape
I = np.ones((O,1))
Xb = np.concatenate((I,X),axis=1)
Xbb = np.matmul(Xb, best_V)
F = 1/(1+np.exp(-Xbb))
Fb = np.concatenate((I,F),axis=1)
Fbb = np.matmul(Fb, best_W)
G = 1/(1+np.exp(-Fbb))

Ycheck = []
for j in range(0,O):
    max_val = 0
    max_val = max(G[j][0],G[j][1],G[j][2])
    if G[j][0]==max_val:
        max_val=0
    elif G[j][1]==max_val:
        max_val=1
    else:
        max_val=2
    Ycheck.append(max_val)
#List of X1 and X2 where y=0
x10 =[]
x20 = []
#List of X1 and X2 where y=1
x11 =[]
x21 = []
#List of X1 and X2 where y=2
x12 =[]
x22 = []
for i in range(len(Ycheck)):
    if(Ycheck[i]==0):
        x10.append(X[i][0])
        x20.append(X[i][1])
    elif(Ycheck[i]==1):
        x11.append(X[i][0])
        x21.append(X[i][1])
    else:
        x12.append(X[i][0])
        x22.append(X[i][1])
plt.plot(x10, x20, "ro", label = "y=0")
plt.plot(x11, x21,  "bo", label = "y=1")
plt.plot(x12, x22,  "go", label = "y=2")
plt.axis([0, 10, 0, 10])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Predicted output")
plt.legend()
plt.show()
