#GMM algorithm

import numpy as np

import math as math

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


##initialize parameters teta

J = 2 #nb of clusters, arbitrary

N = 3 #nb of dimensions


phi = [1/J,1/J]

mu = np.array([[0,3],
                [0,3],
                [0,3]])

covariance_matrix = np.array(
                [[[0.7,0,0],
                [0,0.7,0],
                [0,0,0.7]],

                [[1, 0,0],
                [0, 1,0],
                 [0, 0,1]]])


n_samples = 300

# generate random sample, two components
np.random.seed(0)

# generate spherical data centered
shifted_gaussian = np.dot(np.random.randn(n_samples, N), covariance_matrix[0,:,:]) + mu[:,0]


#print(shifted_gaussian.shape)
# generate zero centered stretched Gaussian data
stretched_gaussian = np.dot(np.random.randn(n_samples, N), covariance_matrix[1,:,:]) + mu[:,1]

#print(stretched_gaussian.shape)
# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])


##EM algorithm

#E-step

def e_step(X_train, phi, mu, covariance):
    I = len(X_train)
    Wj = np.zeros((I,J))

    for i in range(I):
        for j in range(J):
            sumInf = 0
            for k in range (J):
                fracInf = (1/(2*math.pi)**N/2)*np.linalg.det(covariance[k,:,:])**0.5
                soustractionInf = np.reshape((X_train[i]-mu[:,k]),[N,1])
                transposeInf = np.transpose(soustractionInf)
                invCovarianceInf = np.linalg.inv(covariance[k,:,:])

                expInf = np.exp(-0.5*(transposeInf.dot(invCovarianceInf)).dot(soustractionInf))*phi[k]

                sumInf += fracInf*expInf


            detCovariance = np.linalg.det(covariance[j,:,:])
            fracSup = 1/(2*math.pi*detCovariance**0.5)
            transposeSup = np.transpose(X_train[i] - mu[:,j])
            invCovarianceSup = np.linalg.inv(covariance[j,:,:])
            soustractionSup = np.reshape((X_train[i]-mu[:,j]),[N,1])
            expSup = np.exp(-0.5*(transposeSup.dot(invCovarianceSup)).dot(soustractionSup)*phi[j])

            Wj[i][j] = (fracSup*expSup)/sumInf

    return Wj

#M-step
def m_step(Wj,X_train):

    #initialization
    I = len(X_train)
    mu = np.zeros((N,J))
    covariance = np.zeros((J,N,N))
    phi = np.zeros([1,J])
    sumWj = np.sum(Wj) #Sum of each column of Wj

    for j in range(J):

        #we compute mu-j
        sumMu = 0
        for i in range(I):
            sumMu += Wj[i][j]*X_train[i]
        mu[:,j] = sumMu / sumWj

        #we compute phi-j
        phi[:,j] = sumWj/I

        #we compute the covariance-j
        sumCovariance = 0
        substraction = X_train[i]-mu[:,j]
        for i in range(I):
            transpose = np.transpose(substraction)
            sumCovariance += Wj[i][j]*(substraction).dot(transpose)
        covariance[j,:,:] = sumCovariance/sumWj
    return(phi, mu, covariance)


##Find optimal parameters teta*
n = 0
while(n<=1):
    print(n)
    Wj = e_step(X_train,phi,mu,covariance_matrix)
    phi, mu, covariance_matrix = m_step(Wj, X_train)
    n +=1

print(Wj)
print(covariance_matrix)

##Find cluster Ychapeau*
Y_pred = np.argmax(Wj, axis=1)
print(Y_pred)

##Plot the clustering result
def show_plot(Y_pred):
    fig1=plt.figure()
    ax=Axes3D(fig1)
    I = len(X_train)

    for i in range(I):
        if (Y_pred[i]==0):
            ax.scatter(X_train[i, 0], X_train[i, 1],  X_train[i, 2], color='red')
        else:
            ax.scatter(X_train[i, 0], X_train[i, 1],  X_train[i, 2], color='blue')

    plt.show()

show_plot(Y_pred)