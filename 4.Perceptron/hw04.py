# -*- coding: utf-8 -*-
"""
File:   hw04.py
Author: 
Date:   
Desc:   
    
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from collections import Counter
from sklearn.externals import joblib
from sklearn.cluster import KMeans
import random

def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	#return 1.0 if activation >= 0.0 else 0.0
	return 1 if activation >=0 else 0.0
	
	
def train_weights1(data, l_rate, n_epoch,weights):
	for epoch in range(n_epoch):
		for row in data:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	return weights

	
#1
data1 = np.load('dataSet1.npy')
data2 = np.load('dataSet2.npy')
data3 = np.load('dataSet3.npy')



y1=data1[:,2]
x1=data1[:,0:2]
idx = y1.argsort()[::-1]   
y1= y1[idx]
x1 = x1[idx]
c1=0
c2=1

add1=np.ones((data1[:,0].size,1))
x1=np.hstack((add1,x1))
alpha = 10 #0.0003
epoch = 50 #50
w1=[-0.5,0,1]
w2=[-3,0,2]
w1= train_weights1(data1, alpha, epoch,w1)
w2= train_weights1(data1, alpha, epoch,w2)
w1=[-0.5,0,1]
w2=[-3,0,2]
w3=[1,-1,1]

res1=w1[0]*1+w1[1]*x1[:,1]+w1[2]*x1[:,2]
for i in range (res1.size):
	if res1[i]>0:
		res1[i]=1
	else:
		res1[i]=0
res2=w2[0]*1+w2[1]*x1[:,1]+w2[2]*x1[:,2]
for i in range (res2.size):
	if res2[i]>0:
		res2[i]=1
	else:
		res2[i]=0

res3=w3[0]*1+res1*w3[1]+res2*w3[2]

print(res3==y1)


x1plot=np.linspace(-1, 2.5, num=10)

plt.plot(x1plot, (x1plot*w1[1]+w1[0])/-w1[2], label="w1")
plt.plot(x1plot, (x1plot*w2[1]+w2[0])/-w2[2], label="w2")
plt.legend()
for i in range(idx.size):
	if (y1[i]==1):
		plt.scatter(x1[i,1],x1[i,2],c='r',label=1 if c1==0 else "", marker='s')
		c1=c1+1
	else:
		plt.scatter(x1[i,1],x1[i,2], c='b', label=0 if c2==0 else "", marker='x')
		c2=c2+1
		
plt.show()

w1=[-0.5,1,0]
w2=[-1.5,1,0]
w3=[0,1,-1]
res1=w1[0]*1+w1[1]*x1[:,1]+w1[2]*x1[:,2]
for i in range (res1.size):
	if res1[i]>0:
		res1[i]=1
	else:
		res1[i]=0
res2=w2[0]*1+w2[1]*x1[:,1]+w2[2]*x1[:,2]
for i in range (res2.size):
	if res2[i]>0:
		res2[i]=1
	else:
		res2[i]=0
res3=res1*1+res2*-1
print(res3==y1)

x1plot=np.linspace(-1, 2.5, num=10)
y1plot =np.linspace(-1,2.5,num=10)
plt.plot(0.5*np.ones(10), y1plot, label="w1")
plt.plot(1.5*np.ones(10)	, y1plot, label="w2")
plt.legend()
for i in range(idx.size):
	if (y1[i]==1):
		plt.scatter(x1[i,1],x1[i,2],c='r',label=1 if c1==0 else "", marker='s')
		c1=c1+1
	else:
		plt.scatter(x1[i,1],x1[i,2], c='b', label=0 if c2==0 else "", marker='x')
		c2=c2+1
		
plt.show()

#2

y2=data2[:,2]
x2=data2[:,0:2]
idx2 = y2.argsort()[::-1]   
y2= y2[idx2]
x2 = x2[idx2]
c1=0
c2=0
c3=0
alpha = 10 #0.0003
epoch = 50 #50
w1=[0.4,1.8,2]
w2=[0.8,-4.5,3]
w3=[0,1,0]
w4=[0,0,1]
#w1= train_weights1(data1, alpha, epoch,w1)
#w2= train_weights1(data1, alpha, epoch,w2)
#w1=[-0.5,0.0001,1]
#w2=[-3,0.00001,2]

res1=w1[0]*1+w1[1]*x2[:,0]+w1[2]*x2[:,1]
for i in range (res1.size):
	if res1[i]>0:
		res1[i]=1
	else:
		res1[i]=0

res2=w2[0]*1+w2[1]*x2[:,0]+w2[2]*x2[:,1]
for i in range (res2.size):
	if res2[i]>0:
		res2[i]=1
	else:
		res2[i]=0
		
output1=res1

output2=res2

output3=[]
for i in range(output1.size):
	if output1[i]==0:
		output3=np.append(output3,0)
	elif (output1[i]==1 and output2[i]==1):
		output3=np.append(output3,1)
	elif (output1[i]==1 and output2[i]==0):
		output3=np.append(output3,2)
print(output3==y2)


		
x1plot=np.linspace(-1.5, 1.5, num=10)
print(w1)
print(w2)
plt.plot(x1plot, (x1plot*w1[1]+w1[0])/-w1[2], label="w1")
plt.plot(x1plot, (x1plot*w2[1]+w2[0])/-w2[2], label="w2")
plt.legend()
for i in range(idx2.size):
	if (y2[i]==2):
		plt.scatter(x2[i,0],x2[i,1],c='m',label=2 if c3==0 else "", marker='s')
		c3=c3+1
	elif (y2[i]==1):
		plt.scatter(x2[i,0],x2[i,1],c='c',label=1 if c2==0 else "", marker='s')
		c2=c2+1
	else:
		plt.scatter(x2[i,0],x2[i,1], c='y', label=0 if c1==0 else "", marker='s')
		c1=c1+1
plt.show()

w1=[-0.4,1,0]
w2=[0.8,4.5,3]
#w1= train_weights1(data1, alpha, epoch,w1)
#w2= train_weights1(data1, alpha, epoch,w2)
#w1=[-0.5,0.0001,1]
#w2=[-3,0.00001,2]

res1=w1[0]*1+w1[1]*x2[:,0]+w1[2]*x2[:,1]
for i in range (res1.size):
	if res1[i]>0:
		res1[i]=1
	else:
		res1[i]=0

res2=w2[0]*1+w2[1]*x2[:,0]+w2[2]*x2[:,1]
for i in range (res2.size):
	if res2[i]>0:
		res2[i]=1
	else:
		res2[i]=0
		
output1=res1

output2=res2

output3=[]
for i in range(output1.size):
	if output1[i]==1:
		output3=np.append(output3,2)
	elif (output1[i]==0 and output2[i]==1):
		output3=np.append(output3,1)
	elif (output1[i]==0 and output2[i]==0):
		output3=np.append(output3,0)

print(output3==y2)



x1plot=np.linspace(-1.5, 1.5, num=10)
print(w1)
print(w2)
plt.plot(x1plot, 0.4*np.ones(10), label="w1")
plt.plot(x1plot, (x1plot*w2[1]+w2[0])/-w2[2], label="w2")
plt.legend()
for i in range(idx2.size):
	if (y2[i]==2):
		plt.scatter(x2[i,0],x2[i,1],c='m',label=2 if c3==0 else "", marker='s')
		c3=c3+1
	elif (y2[i]==1):
		plt.scatter(x2[i,0],x2[i,1],c='c',label=1 if c2==0 else "", marker='s')
		c2=c2+1
	else:
		plt.scatter(x2[i,0],x2[i,1], c='y', label=0 if c1==0 else "", marker='s')
		c1=c1+1
plt.show()

#3
y3=data3[:,2]
x3=data3[:,0:2]
idx3 = y3.argsort()[::-1]   
y3= y3[idx3]
x3 = x3[idx3]

c1=0
c2=0
c3=0

w1=[2,-1.3,1]
w2=[0,-1.3,1]
w3=[-2,-1.3,1]
w4=[1,-1,1,-1]
res1=w1[0]*1+w1[1]*x3[:,0]+w1[2]*x3[:,1]
for i in range (res1.size):
	if res1[i]>0:
		res1[i]=1
	else:
		res1[i]=0
res2=w2[0]*1+w2[1]*x3[:,0]+w2[2]*x3[:,1]
for i in range (res2.size):
	if res2[i]>0:
		res2[i]=1
	else:
		res2[i]=0
		
res3=w3[0]*1+w3[1]*x3[:,0]+w3[2]*x3[:,1]
for i in range (res3.size):
	if res3[i]>0:
		res3[i]=1
	else:
		res3[i]=0
		
res4=w4[0]+w4[1]*res1+w4[2]*res2+w4[3]*res3
print(res4==y3)


x1plot=np.linspace(0, 5, num=10)
plt.plot(x1plot, (x1plot*w1[1]+w1[0])/-w1[2], label="w1")
plt.plot(x1plot, (x1plot*w2[1]+w2[0])/-w2[2], label="w2")
plt.plot(x1plot, (x1plot*w3[1]+w3[0])/-w3[2], label="w3")
plt.legend()
for i in range(idx3.size):
	if (y3[i]==1):
		plt.scatter(x3[i,0],x3[i,1],c='m',label=1 if c2==0 else "", marker='s')
		c3=c3+1
	elif (y3[i]==0):
		plt.scatter(x3[i,0],x3[i,1],c='c',label=0 if c1==0 else "", marker='s')
		c2=c2+1

plt.show()

c1=0
c2=0
c3=0
w1=[-1.3,1,0.001]
w2=[-2.8,1,0.1]
w3=[-3.7,1,0.1]
w4=[0,1,-1,1]
res1=w1[0]*1+w1[1]*x3[:,0]+w1[2]*x3[:,1]
for i in range (res1.size):
	if res1[i]>0:
		res1[i]=1
	else:
		res1[i]=0
res2=w2[0]*1+w2[1]*x3[:,0]+w2[2]*x3[:,1]
for i in range (res2.size):
	if res2[i]>0:
		res2[i]=1
	else:
		res2[i]=0
		
res3=w3[0]*1+w3[1]*x3[:,0]+w3[2]*x3[:,1]
for i in range (res3.size):
	if res3[i]>0:
		res3[i]=1
	else:
		res3[i]=0
		
res4=w4[0]+w4[1]*res1+w4[2]*res2+w4[3]*res3

print(res4==y3)
x1plot=np.linspace(0, 5, num=10)
x1plot=np.linspace(0, 5, num=10)
plt.plot(x1plot, (x1plot*w1[1]+w1[0])/-w1[2], label="w1")
plt.plot(x1plot, (x1plot*w2[1]+w2[0])/-w2[2], label="w2")
plt.plot(x1plot, (x1plot*w3[1]+w3[0])/-w3[2], label="w3")
plt.axis([0,5,-1,6])
plt.legend()
for i in range(idx3.size):
	if (y3[i]==1):
		plt.scatter(x3[i,0],x3[i,1],c='m',label=1 if c2==0 else "", marker='s')
		c3=c3+1
	elif (y3[i]==0):
		plt.scatter(x3[i,0],x3[i,1],c='c',label=0 if c1==0 else "", marker='s')
		c2=c2+1

plt.show()