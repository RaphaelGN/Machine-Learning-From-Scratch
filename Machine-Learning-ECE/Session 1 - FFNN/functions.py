import numpy as np
import matplotlib.pyplot as plt


def arrondi(x):
    return (x>0.5).astype(int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def activation(x):
    return sigmoid(x)

def fwp(X,V,W):
    I=len(X)
    # Forward Propagation

    #Computing Xb
    ones=np.ones((I, 1))
    Xb=np.concatenate((ones, X), axis=1)

    #Computing Xbb
    Xbb=np.dot(Xb,V)

    #Computing F
    F=np.apply_along_axis(activation, 0, Xbb)

    #Computing Fb
    Fb=np.concatenate((ones, F), axis=1)

    #Computing Fbb
    Fbb=np.dot(Fb,W)

    #Computing G
    G=np.apply_along_axis(activation, 0, Fbb)

    return G,F,Fb,Xb

def bp(V,W,Y,G,F,Fb,Xb,J,K,N,av,aw):
    I=len(Y)
    #BACK Propagation
    H=(G-Y)*G*(1-G)

    #Computing the new V
    for n in range(0,N+1):
        for k in range(0,K):
            #for V[n][k]
            dEdV=0
            for i in range(0,I):
                for j in range(0,J):
                    dEdV += H[i][j] * W[k][j] * F[i][k] * (1 - F[i][k]) * Xb[i][n]

            V[n][k] -= aw * dEdV

    #Computing the new W
    for k in range(0,K+1):
        for j in range(0,J):
            #for W[k][j]
            dEdW=0
            for i in range(0,I):
                dEdW += H[i][j] *Fb[i][k]
            #(G-Y)*G*(1-G)

            W[k][j] -= aw * dEdW

    return V,W

def error(Y,Yp,J):
    I=len(Y)
    #Yp=np.apply_along_axis(arrondi, 0, Yp)
    E=0
    for i in range(0,I):
        for j in range(0,J):
            predicted=Yp[i][j]
            target=Y[i][j]
            E+=np.square(predicted-target)
    E/=2

    # E=np.apply_along_axis(abs, 0, Yp-Y)
    # E/=2


    return E

def getData(file):
    data = np.loadtxt(fname = "data.txt")
    X=data[:,:2]
    YData=data[:,2:]


    #Formatting Y
    Y=[]
    YUnique=np.unique(YData)

    for y in YData:
        Y.append((YUnique==y[0]).astype(int))

    Y=np.asarray(Y)

    return X,Y,YUnique

