import numpy as np
import matplotlib.pyplot as plt

#Getting X and Y
X = np.loadtxt(fname = "data_pca.txt")



I=len(X)
N=len(X[0])
K=3

#computing mu
mu=np.sum(X,axis=0)/I

#computing dataBar
Xb=X-mu

#computing sigma COVARIANCE DE X
sigma=np.zeros((N,N))
for x in Xb:
    sigma += x * x.reshape((N,1))
sigma/=I

#computing the eigenvectors
eig_vals, eig_vecs = np.linalg.eig(sigma)

#sorting eigens
indexMax=np.argmax(eig_vals)
l=eig_vals[indexMax]
u=eig_vecs[indexMax]

#Yb=Xb*u
#temp = Xb.dot(u)
# Yb=temp*u
# Y=Yb#+mu

#print(Y)

plt.scatter(X[:,0], X[:,1], marker="x")

# plt.scatter(Y, np.zeros(len(Y)), marker="*")
plt.show()
