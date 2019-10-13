# -*- coding: utf-8 -*-
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
mean1 = [100, 0]
cov1 = [[500, 0], [0, 100]]
mean2 = [250, 0]
cov2 = [[150, 0], [0, 750]]
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
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print("original eigen values")
print(eigen_vals)
print("original eigen vectors")
print(eigen_vecs)
idx = np.argsort(eigen_vals)[::-1]
evecs = eigen_vecs[:,idx]
evals = eigen_vals[idx]
print(evecs)
print(evals)
ZZdeco=np.dot(ZZ,evecs)
Z_pca=ZZdeco/np.sqrt(evals+1e-5)

for i in range(t.size):
	if (t[i]==0):
		plt.scatter(ZZ[i,0],ZZ[i,1], c='r', label=0 if c1==0 else "", marker='s')
		c1=c1+1
	else:
		plt.scatter(ZZ[i,0],ZZ[i,1], c='b', label=1 if c2==0 else "", marker='x')
		c2=c2+1
plt.legend(loc='upper left');		
plt.show()
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
print("original covariance generated from data standard scaled")
print(np.cov(ZZ.T))
print("covariance of decorrelated data via PCA")
print(np.cov((ZZ.dot(eigen_vecs)).T))
print("whitened covariance")
print(np.cov(Z_pca.T))

#METHOD2 from https://www.kdnuggets.com/2016/03/must-know-tips-deep-learning-part-1.html for testing purposes
#about this. i am unsure why did my whole dataset rotate by the vertical axis of the  graph. It seems like it has a bias on one of the principal components
cov = np.cov(ZZ.T)
U,S,V = np.linalg.svd(cov)
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