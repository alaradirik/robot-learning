from gaussPDF import *
import numpy as np
import sys



def kMeans(X, K, maxIters = 30):
    centroids = X[np.random.choice(np.arange(len(X)), K), :]
    for i in range(maxIters):
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        centroids = [X[C == k].mean(axis = 0) for k in range(K)]
    return np.array(centroids) , C

def EM_init(Data, nbStates):
    nbVar, nbData = np.shape(Data)
    Priors = np.ndarray(shape = (1, nbStates))
    Sigma = np.ndarray(shape = (nbVar, nbVar, nbStates))
    Centers, Data_id = kMeans(np.transpose(Data), nbStates)
    Mu = np.transpose(Centers)
    for i in range (0,nbStates):
        idtmp = np.nonzero(Data_id==i)
        idtmp = list(idtmp)
        idtmp = np.reshape(idtmp,(np.size(idtmp)))
        Priors[0,i] = np.size(idtmp)
        a = np.concatenate((Data[:, idtmp],Data[:, idtmp]), axis = 1)
        Sigma[:,:,i] = np.cov(a)
        Sigma[:,:,i] = Sigma[:,:,i] + 0.00001 * np.diag(np.diag(np.ones((nbVar,nbVar))))
    Priors = Priors / nbData
    return (Priors, Mu, Sigma)
    
def EM(Data, Priors0, Mu0, Sigma0):
    realmax = sys.float_info[0]
    realmin = sys.float_info[3]
    loglik_threshold = 1e-10
    nbVar, nbData = np.shape(Data)
    nbStates = np.size(Priors0)
    loglik_old = -realmax
    nbStep = 0
    Mu = Mu0
    Sigma = Sigma0
    Priors = Priors0

    Pix = np.ndarray(shape = (nbStates, nbData))
    Pxi = np.ndarray(shape = (nbData, nbStates))
    while 1:
        for i in range (0,nbStates):
            Pxi[:,i] = gaussPDF(Data,Mu[:,i],Sigma[:,:,i])

        Pix_tmp = np.multiply(np.tile(Priors, (nbData, 1)),Pxi)
        Pix = np.divide(Pix_tmp,np.tile(np.reshape(np.sum(Pix_tmp,1), (nbData, 1)), (1, nbStates)))
        E = np.sum(Pix, 0)
        Priors = np.reshape(Priors, (nbStates))
        for i in range (0,nbStates):
            Priors[i] = E[i]/nbData
            Mu[:,i] = np.dot(Data,Pix[:,i])/E[i]
            Data_tmp1 = Data - np.tile(np.reshape(Mu[:,i], (nbVar, 1)), (1,nbData))
            a = np.transpose(Pix[:, i])
            b = np.reshape(a, (1, nbData))
            c = np.tile(b, (nbVar, 1))
            d = c*Data_tmp1
            e = np.transpose(Data_tmp1)
            f = np.dot(d,e)
            Sigma[:,:,i] = f/E[i]
            Sigma[:,:,i] = Sigma[:,:,i] + 0.00001 * np.diag(np.diag(np.ones((nbVar,nbVar))))

        for i in range (0,nbStates):
            Pxi[:,i] = gaussPDF(Data,Mu[:,i],Sigma[:,:,i])
        F = np.dot(Pxi,np.transpose(Priors))
        indexes = np.nonzero(F<realmin)
        indexes = list(indexes)
        indexes = np.reshape(indexes,np.size(indexes))
        F[indexes] = realmin
        F = np.reshape(F, (nbData, 1))
        loglik = np.mean(np.log10(F), 0)
        if np.absolute((loglik/loglik_old)-1)<loglik_threshold:
            break
        loglik_old = loglik
        nbStep = nbStep+1
    return(Priors,Mu,Sigma, Pix)