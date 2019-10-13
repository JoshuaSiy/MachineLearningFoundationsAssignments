# -*- coding: utf-8 -*-
"""
File:   hw03.py
Author: Joshua Siy	
Date:   Oct 22 2018
Desc:   I did my best. By the way to make the whitened covariance  into a diagonal line of [1,0;0,1] just remove the (+1e-5) this was used just in case the eigenvalue =0
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from collections import Counter
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

fig = plt.figure()
ax1 = fig.add_subplot(111)
mean1 = [0, 0]
cov1 = [[10, 0], [0, 60]]
mean2 = [10, 0]
cov2 = [[10, 0], [0, 60]]
Xa1,Xa2 = np.random.multivariate_normal(mean1, cov1, 100).T
A=np.column_stack((Xa1,Xa2))
t= np.ones([1,100])
Xb1,Xb2 = np.random.multivariate_normal(mean2, cov2, 100).T
B=np.column_stack((Xb1,Xb2))
t= np.hstack((t,np.zeros([1,100])))
Z=np.vstack((A,B))
Z=Z-np.mean(Z,axis=0)
t=t.T
print(t.shape)
colors = ['r','b']
markers = ['s', 'x']
c1=0
c2=0
#Dr.Zare's lecture notes

ZZ=Z
cov_mat = np.cov(ZZ.T)
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
print("original eigen values")
print(eigen_vals)
print("original eigen vectors")
print(eigen_vecs)
idx = eigen_vals.argsort()[::-1]   
eigen_vals = eigen_vals[idx]
eigen_vecs = eigen_vecs[:,idx]
for i in range(t.size):
	if (t[i]==0):
		plt.scatter(ZZ[i,0],ZZ[i,1], c='r', label=0 if c1==0 else "", marker='s')
		c1=c1+1
	else:
		plt.scatter(ZZ[i,0],ZZ[i,1], c='b', label=1 if c2==0 else "", marker='x')
		c2=c2+1
plt.legend(loc='upper left');		
plt.show()
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)
print(eigen_vecs)
print('Eigenvalues in descending order:')
for i in eigen_pairs:
    print(i[0])
ZZdeco=np.dot(ZZ,eigen_vecs)#decorrelation matrix
Z_pca = ZZdeco/np.sqrt(eigen_vals+1e-5) 

c1=0
c2=0
print("plot of the decorrelated values under the lecture algorithm")
for i in range(t.size):
	if (t[i]==0):
		plt.scatter(ZZdeco[i,0],ZZdeco[i,1], c='r', label=0 if c1==0 else "", marker='s')
		c1=c1+1
	else:
		plt.scatter(ZZdeco[i,0],ZZdeco[i,1], c='b', label=1 if c2==0 else "", marker='x')
		c2=c2+1
plt.legend(loc='upper left');		
plt.show()
c1=0
c2=0
print("plot of the whitened values under the lecture algorithm")
for i in range(t.size):
	if (t[i]==0):
		plt.scatter(Z_pca[i,0],Z_pca[i,1], c='r', label=0 if c1==0 else "", marker='s')
		c1=c1+1
	else:
		plt.scatter(Z_pca[i,0],Z_pca[i,1], c='b', label=1 if c2==0 else "", marker='x')
		c2=c2+1
plt.legend(loc='upper left');		
plt.show()
print("Results from using the Lecture Algorithm")
print("original covariance generated from data")
print(np.cov(Z.T))
print("covariance of decorrelated data via PCA")
print(np.cov(ZZdeco.T))
print("whitened covariance")
print(np.cov(Z_pca.T))


"""

#this is used to make some comparisons using svd for eigenvector calculation. However for this assignment, the class prefers to use the eigen decomposition than the Single Value decomposition
#My findings tell that SVD produces are more stable result than eigen value decomposition. The results of using SVD matches the PCA library which tells me that SVD is more accurate. 
#However the only difference i see is that the plot of this is just inverted from the plot made above. 


#METHOD2 from https://www.kdnuggets.com/2016/03/must-know-tips-deep-learning-part-1.html for testing purposes
#about this. i am unsure why did my whole dataset rotate by the vertical axis of the  graph. It seems like it has a bias on one of the principal components
cov = np.cov(ZZ.T)
U,S,V = np.linalg.svd(cov)
print(U)
Xrot = np.dot(ZZ,U) #decorrelation U = eigenvector
c1=0
c2=0
print("plot of the decorrelated values under the KDnuggets website algorithm")
for i in range(t.size):
	if (t[i]==0):
		plt.scatter(Xrot[i,0],Xrot[i,1], c='r', label=0 if c1==0 else "", marker='s')
		c1=c1+1
	elif(t[i]==1):
		plt.scatter(Xrot[i,0],Xrot[i,1], c='b', label=1 if c2==0 else "", marker='x')
		c2=c2+1
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
Xwhite = Xrot / np.sqrt(S + 1e-5) #whitening S=lambda = eigenvalue the usage of the 1e-5 value is just to prevent dividing by zero.
c1=0
c2=0
print("plot of the whitened values under the KDnuggets website algorithm")
for i in range(t.size):
	if (t[i]==0):
		plt.scatter(Xwhite[i,0],Xwhite[i,1], c='r', label=0 if c1==0 else "", marker='s')
		c1=c1+1
	elif(t[i]==1):
		plt.scatter(Xwhite[i,0],Xwhite[i,1], c='b', label=1 if c2==0 else "", marker='x')
		c2=c2+1
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
print(Xwhite.shape)
PCAcov=np.cov(Xwhite.T)
print("Results from Kdnuggets")
print("original covariance")
print(np.cov(Z.T))
print("original covariance generated from data standard scaled")
print(np.cov(ZZ.T))
print("covariance of decorrelated data via PCA from KD")
print(np.cov(Xrot.T))
print("whitened covariance from KD")
print(np.cov(Xwhite.T))
#testing purposes
#plt.scatter(Xwhite[(t==0),0],Xwhite[(t==0),1], c='r', label=0, marker='s')
#plt.scatter(Xwhite[(t==1),0],Xwhite[(t==1),1], c='b', label=1, marker='x')
#plt.xlabel('PC 1')
#plt.ylabel('PC 2')
#plt.legend(loc='lower left')
#plt.show()

"""
"""
#This was used to check the data i developed. In my experiments. The results is that KD nuggets' atleast is much more accurate to the results obtained in the PCA library
#The reason to its accuracy in plotting was the generation of eigenvectors compared between eig or eigh against svd. 
#The difference between Single Value Decomposition and Eigen Decomposition


#using PCA library for testing purposes
scikit_pca = PCA(n_components = 2, whiten=True)
scikit_pca1=PCA(n_components = 2, whiten=False)
X_spca = scikit_pca.fit_transform(ZZ)
c1=0
c2=0
print("plot of the PCA library.both whitened and decorrelated")
for i in range(t.size):
	if (t[i]==0):
		plt.scatter(X_spca[i,0],X_spca[i,1], c='r', label=0 if c1==0 else "", marker='s')
		c1=c1+1
	elif(t[i]==1):
		plt.scatter(X_spca[i,0],X_spca[i,1], c='b', label=1 if c2==0 else "", marker='x')
		c2=c2+1
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
PCAcov_spca=np.cov(X_spca.T)

X_spca1 = scikit_pca1.fit_transform(ZZ)
PCAcov1_spca=np.cov(X_spca1.T)
print("Using the PCA library")
print("original covariance")
print(np.cov(Z.T))
print("original covariance generated from data standard scaled")
print(np.cov(ZZ.T))
print("From library, the non-whitened covariance")
print(PCAcov1_spca)
print("From library, the whitened covariance")
print(PCAcov_spca)

"""


"""

print(eigen_vals)
print(U)
print(V)
print(S)
print(eigen_vecs.T)
print(w)
#What i learned is that the PCA library's whitening is dependent on data  where the algorithm provided in KD nuggets is biased on the right hand component, while Dr. Zare's code is biased on the data on the left side. While PCA functions dependently which could probably result to Dr. Zare's algorithm or the one from KD nugget's 
"""



