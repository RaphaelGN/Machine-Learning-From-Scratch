import numpy as np
import matplotlib.pyplot as plt

def updateMu(mu,XList,clusters):
    newMu=mu
    counter=np.ones((len(mu),1))
    I=len(XList)

    for i in range(I): #for each point
        point=XList[i]
        cNumber= clusters[i]

        newMu[cNumber]+=point
        counter[cNumber]+=1

    newMu /= counter

    return newMu

def closerMu(mu, x):
    return np.argmin(np.sum(np.square(mu-x),axis=1))

#Getting X and Y
data = np.loadtxt(fname = "data_kmeans.txt")

I=len(data)
N=len(data[0])
K=3

plt.scatter(data[:,0], data[:,1], marker="x")

mu=np.random.uniform(np.amin(data),np.amax(data),(K,N))
muOLD=mu*2

while(np.sum(abs(mu-muOLD))!=0):
    clusters=[]
    for x in data:
        clusters.append(closerMu(mu,x))

    muOLD=np.copy(mu)
    mu=updateMu(mu,data,clusters)


for m in range(len(mu)):
    print(np.round(mu[m],1)," ",m)
print()

X_Test=np.random.uniform(np.amin(data),np.amax(data),(10,N))
clusters_Test=[]
for x in X_Test:
    c=closerMu(mu,x)
    clusters_Test.append(c)
    print(np.round(x,1)," ",c)


plt.scatter(X_Test[:,0],X_Test[:,1],marker="*")
plt.scatter(mu[:,0],mu[:,1],marker="o")
plt.show()

