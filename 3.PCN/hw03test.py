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
mean1 = [0, 0]
cov1 = [[13, 0], [0, 28]]
mean2 = [15, 0]
cov2 = [[12, 0], [0, 45]]
fig = plt.figure()
ax1 = fig.add_subplot(111)
W1,W2 = np.random.multivariate_normal(mean1, cov1, 100).T
A=np.column_stack((W1,W2))
t= np.ones([1,100])
V1,V2 = np.random.multivariate_normal(mean2, cov2, 100).T
B=np.column_stack((V1,V2))
t    = np.hstack((t,np.zeros([1,100])))
Z=np.vstack((A,B))
print(Z.shape)
t=t.T
c1=0
c2=0
for i in range(t.size):
	if (t[i]==0):
		plt.scatter(Z[i,0],Z[i,1], c='r', label=0 if c1==0 else "", marker='s')
		c1=c1+1
	else:
		plt.scatter(Z[i,0],Z[i,1], c='b', label=1 if c2==0 else "", marker='x')
		c2=c2+1
plt.legend(loc='upper left');		
plt.show()
		#sc      = StandardScaler()
#Z = sc.fit_transform(Z)
#cov = np.dot(Z.T, Z) / Z.shape[0]
#print(cov)
#wait=input("")
cov = np.cov(Z.T)
U,S,V = np.linalg.svd(cov)
eigen_vals, eigen_vecs = np.linalg.eig(cov)
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)
print(S)
Xrot = np.dot(Z, U)
Xwhite = Xrot / np.sqrt(S + 1e-5)
print(Xwhite[0])
colors = ['r','b']
markers = ['s', 'x']
#for l,c,m in zip(np.unique(t), colors, markers):
#	plt.scatter(Xwhite[t==l,0],Xwhite[t==l,1], c=c, label=l, marker=m)
plt.scatter(Xwhite[0:99,0],Xwhite[0:99,1], c='r', label=0, marker='s')
plt.scatter(Xwhite[100:199,0],Xwhite[100:199,1], c='b', label=1, marker='x')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
print(Xwhite.shape)
"""
File:   hw03.py
Author: 
Date:   
Desc:   
"""