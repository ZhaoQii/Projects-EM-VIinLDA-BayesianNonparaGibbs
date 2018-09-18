# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:40:48 2017

@author: Qi Zhao qz2316
"""
from matplotlib import style
from scipy.stats import poisson
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp

style.use('ggplot')

# Problem 1
# Read the data
X = pd.read_csv('/Users/ap/Dropbox/2017FALL/EECS E6720BayesianModelforML/HW4/x.csv',

def L_cal(X, K, pi, lamb, phi):
    temp = np.tile(np.matrix(np.log(pi)-lamb), (X.shape[0], 1))
    temp = temp + np.outer(np.matrix(X), np.matrix(np.log(lamb)))
    temp = temp - np.tile(np.log(sp.special.factorial(X)), (1, K))
    return(np.sum(np.multiply(phi, temp)) - np.sum(np.multiply(phi, np.log(phi)))) #- np.sum(np.log(sp.misc.factorial(X))))


def EM(K, T=50):
    
    # Initilization
    #pi = np.random.dirichlet(np.ones(K),size=1)
    pi = np.ones(K)/K
    #lamb = np.ones(K)
    lamb = np.random.gamma(K, size=K)
    L = np.zeros(T)
    
    for t in range(T):
        
        # E-Step:
        poi_like = [X.apply(poisson.pmf, mu=l) for l in lamb]
        poi_like = pd.concat(poi_like, axis = 1)
        poi_like = np.matrix(poi_like.as_matrix())
        poi_like = np.multiply(np.tile(pi, (X.shape[1],1)), poi_like)
        poi_like = np.divide(poi_like, np.tile(np.sum(poi_like, 1), (1,K)))
        
        # M-Step:
        nj = np.sum(poi_like, axis=0)
        nj = np.array(nj).ravel()
        pi = nj/np.sum(nj)
        lamb = np.sum(np.multiply(poi_like, np.tile(X.as_matrix(), (1,K))), 0) / nj
        lamb = np.array(lamb).ravel()
        
        # Evaluate Convergence
        L[t] = L_cal(X, K, pi, lamb, phi = poi_like)
    return(L, pi, lamb)
    
# b)
L15, pi15, lamb15 = EM(15)
L3, pi3, lamb3 = EM(3)
L9, pi9, lamb9 = EM(9)
plt.plot(L15[1:], color = 'blue', label = 'K=15')
plt.plot(L3[1:], color = 'red', label = 'K=3')
plt.plot(L9[1:], color = 'green', label = 'K=9')
plt.ylabel('Log Marginal Likelihood')
plt.xlabel('Iterations from 2th to 50th')
plt.xticks([])
plt.legend(loc = 'lower right')
plt.show()

# c)
# Define the function for calculating probabilities
def cal_1c(xhat, lamb, pi):
    return(sp.stats.poisson.pmf(xhat, mu=lamb)*pi)
   
xhat = np.linspace(0, 50, 51)
p3 = np.zeros([len(xhat), 3])
for i in range(len(xhat)):
    for j in range(3):
        p3[i, j] = cal_1c(xhat[i], lamb3[j], pi3[j])
plt.scatter(xhat, p3.argmax(axis=1)+1)
plt.xlabel('Integer from 1 to 50')
plt.ylabel('Most Probable Cluster')
plt.xlim([0, 50])
plt.ylim([0,3.5])
plt.title('K=3')
plt.show()

p9 = np.zeros([len(xhat), 9])
for i in range(len(xhat)):
    for j in range(9):
        p9[i, j] = cal_1c(xhat[i], lamb9[j], pi9[j])
plt.scatter(xhat, p9.argmax(axis=1)+1)
plt.xlabel('Integer from 1 to 50')
plt.ylabel('Most Probable Cluster')
plt.xlim([0, 50])
plt.ylim([0,9.5])
plt.title('K=9')
plt.show()

p15 = np.zeros([len(xhat), 15])
for i in range(len(xhat)):
    for j in range(15):
        p15[i, j] = cal_1c(xhat[i], lamb15[j], pi15[j])
plt.scatter(xhat, p15.argmax(axis=1)+1)
plt.xlabel('Integer from 1 to 50')
plt.ylabel('Most Probable Cluster')
plt.xlim([0, 50])
plt.ylim([0,15.5])
plt.title('K=15')
plt.show()


# Problem 2
  
def cal_obj(phi, a, b, alpha, a0, b0, alpha0, K):
    temp = np.tile(-a/b, (X.shape[0], 1)) + np.outer(X, (sp.special.digamma(a)-np.log(b))) - np.tile(np.log(sp.special.factorial(X)), (1, K))
    temp = temp + np.tile(sp.special.digamma(alpha), (X.shape[0], 1)) - np.ones((X.shape[0], K))*sp.special.digamma(np.sum(alpha))
    obj = np.sum(np.multiply(phi, temp)) + (alpha0-1)*np.sum(sp.special.digamma(alpha)-sp.special.digamma(np.sum(alpha))) + sp.special.gammaln(K*alpha0) - K * sp.special.gammaln(alpha0)
    obj = obj + np.sum(a0*np.log(b0)-sp.special.gammaln(a0)+(a0-1)*(sp.special.digamma(a)-np.log(b))-b0*a/b)
    obj = obj - np.sum(np.multiply((alpha-1),(sp.special.digamma(alpha)-sp.special.digamma(np.sum(alpha)))))+np.sum(sp.special.gammaln(alpha))-sp.special.gammaln(np.sum(alpha))  
    obj = obj - np.sum(np.multiply(phi, np.log(phi))) + np.sum(a-np.log(b)+sp.special.gammaln(a)+np.multiply((1-a),sp.special.digamma(a)))
    #obj = obj + sp.special.gammaln(K*alpha0) - K * sp.special.gammaln(alpha0)
    return(obj)    
    


def VI(K, T=1000):

    # Set hyperparameters
    alpha0 = 1/10
    a0 = 4.5
    b0 = 0.25
    
    # Initialization
    a = np.random.uniform(0,1,K)
    b = np.random.uniform(0,1,K)
    alpha = np.random.uniform(0,0.5,K)
    obj = np.zeros(T)
    
    # Iteration 
    for t in range(T):
        
        # Update q(c)
        phi = np.tile(-a/b, (X.shape[0], 1)) + np.outer(X, (sp.special.digamma(a)-np.log(b)))
        phi = phi + np.tile(sp.special.digamma(alpha), (X.shape[0], 1)) - np.ones((X.shape[0], K))*sp.special.digamma(np.sum(alpha))
        
        phi = np.asmatrix(np.exp(phi))
        phi = np.divide(phi, np.tile(np.sum(phi, 1), (1,K)))
               
        nj = np.sum(phi, 0)
        
        # Update q(\pi)
        alpha = alpha0 + nj
    
        # Update q(\lambda_{j})
        a = a0 + np.dot(np.transpose(np.matrix(X)), phi)
        b = b0 + nj
        
        obj[t] = cal_obj(phi, a, b, alpha, a0, b0, alpha0, K)
    return(obj, a, b, alpha)

# b)
    
obj50, a50, b50, alpha50 = VI(50)
obj15, a15, b15, alpha15 = VI(15)
obj3, a3, b3, alpha3 = VI(3)

plt.plot(obj3[1:], color = 'blue', label = 'K=3')
plt.plot(obj15[1:], color = 'red', label = 'K=15')
plt.plot(obj50[1:], color = 'green', label = 'K=50')
plt.ylabel('Objective Function')
plt.xlabel('Iterations from 2th to 1000th')
plt.xticks([])
plt.legend(loc = 'lower right')
plt.show()

# c)

# Define the function to calculate probabilities
def cal_2c(xhat, a, b, alpha, j):
   cal = np.log(alpha[0,j])-np.log(np.sum(alpha))-np.log(sp.special.factorial(xhat)) + a*np.log(b) - sp.special.gammaln(a) + sp.special.gammaln(xhat+a) - (xhat+a)*np.log(b+1)
   return(np.exp(cal))

xhat = np.linspace(0, 50, 51)
p3 = np.zeros([len(xhat), 3])
for i in range(len(xhat)):
    for j in range(3):
        p3[i, j] = cal_2c(xhat[i], a3[0,j], b3[0, j], alpha3, j)
plt.scatter(xhat, p3.argmax(axis=1)+1)
plt.xlabel('Integer from 1 to 50')
plt.ylabel('Most Probable Cluster')
plt.xlim([0, 50])
plt.ylim([0,3.5])
plt.title('K=3')
plt.show()


p15 = np.zeros([len(xhat), 15])
for i in range(len(xhat)):
    for j in range(15):
        p15[i, j] = cal_2c(xhat[i], a15[0,j], b15[0, j], alpha15, j)
plt.scatter(xhat, p15.argmax(axis=1)+1)
plt.xlabel('Integer from 1 to 50')
plt.ylabel('Most Probable Cluster')
plt.xlim([0, 50])
plt.ylim([0,15.5])
plt.title('K=15')
plt.show()


p50 = np.zeros([len(xhat), 50])
for i in range(len(xhat)):
    for j in range(50):
        p50[i, j] = cal_2c(xhat[i], a50[0,j], b50[0, j], alpha50, j)
plt.scatter(xhat, p50.argmax(axis=1)+1)
plt.xlabel('Integer from 1 to 50')
plt.ylabel('Most Probable Cluster')
plt.xlim([0, 50])
plt.ylim([0,50.5])
plt.title('K=50')
plt.show()



# Problem 3


# a
# Initialization 
alpha0 = 3/4
a0 = 4.5
b0 = 1/4
T = 1000
J = 30
n = X.shape[0]
lamb = np.random.gamma(shape = np.mean(X), scale = b0, size = J)
c = np.random.randint(0, J, size = n)

J_iter = np.zeros(T)
J_iter[0] = J

con1 = alpha0 * b0**a0 / (alpha0+n-1) / sp.special.gamma(a0) * sp.special.gamma(a0+X) / sp.special.factorial(X) / (b0+1)**(a0+X)

top6 = np.zeros((6, T))



def cal_phi(lamb, c, i, X):
    phi = np.zeros(len(lamb))
    nj_i = np.asarray([np.sum(np.delete(c, i) == j) for j in range(len(lamb))])
    njibig0 = np.where(nj_i>0)[0]
    phi[njibig0] = nj_i[njibig0] * sp.stats.poisson.pmf(X.iloc[i], mu=lamb[njibig0]) / (alpha0+n-1)
    phi = np.append(phi, con1.iloc[i])
    phi = phi/np.sum(phi)    
    return(phi, njibig0)


import progressbar
bar = progressbar.ProgressBar()

for t in bar(range(1,T)):

    for i in range(n):
        phi, njibig0 = cal_phi(lamb, c, i, X)
        c[i] = np.random.choice(np.arange(0,len(lamb)+1), p=phi)
        
        if c[i] == len(lamb): # new j
            lamb = np.append(lamb, np.random.gamma(shape = a0+X.iloc[i], scale = b0+1))
    
    n_j = np.bincount(c)
    nbig0 = np.where(n_j>0)[0]
    J = len(nbig0) 
    J_iter[t] = J
    lamb = [np.random.gamma(shape = a0+np.sum(X.iloc[np.where(c==nbig0[j])[0]]), scale = b0+np.sum(c==nbig0[j])) for j in range(J)]
    lamb = np.asarray(lamb).reshape((J,))
    
    if len(n_j)>6:
        top6[:,t]=n_j[n_j.argsort()[-6:][::-1]]
    if len(n_j)<=6:
        top6[0:len(n_j),t] =n_j[n_j.argsort()[::-1]]
    

# b
plt.plot(J_iter)
plt.show()
        

# c      
plt.plot(top6.transpose(), linewidth=0.6)
plt.show()
