# -*- coding: utf-8 -*-
"""
File:   hw02.py
Author: 
Date:   
Desc:   
    
"""


""" =======================  Import dependencies ========================== """

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from collections import Counter

#plt.close('all') #close any open plots

""" =======================  Import DataSet ========================== """


Train_2D = np.loadtxt('2dDataSetforTrain.txt')
Train_7D = np.loadtxt('7dDataSetforTrain.txt')
Train_HS = np.loadtxt('HyperSpectralDataSetforTrain.txt')

labels_2D = Train_2D[:,Train_2D.shape[1]-1]
labels_7D = Train_7D[:,Train_7D.shape[1]-1]
labels_HS = Train_HS[:,Train_HS.shape[1]-1]

Train_2D = np.delete(Train_2D,Train_2D.shape[1]-1,axis = 1)
Train_7D = np.delete(Train_7D,Train_7D.shape[1]-1,axis = 1)
Train_HS = np.delete(Train_HS,Train_HS.shape[1]-1,axis = 1)

Test_2D = np.loadtxt('2dDataSetforTest.txt')
Test_7D = np.loadtxt('7dDataSetforTest.txt')
Test_HS = np.loadtxt('HyperSpectralDataSetforTest.txt')

"""Functions and Definitions"""

def	PGC2C(Train,Classes):#calculating mean covariance and priors
	mu0=np.mean(Train[0], axis=0)
	mu1=np.mean(Train[1], axis=0)
	print(Train[0].shape)
	wait=input("")
	cov0 = np.cov(Train[0].T)
	cov1 = np.cov(Train[1].T)
	pC0 = Train[0].shape[0]/(Train[0].shape[0] + Train[1].shape[0])
	pC1 = Train[1].shape[0]/(Train[0].shape[0] + Train[1].shape[0])
	return mu0,mu1,cov0,cov1,pC0,pC1
def PGC2CD(Train,mu0,mu1):
	vol0=((Train[0]-mu0)**2)
	vol0=vol0.sum(axis=0)/(Train[0][:,1].size)
	covd0=np.diag(vol0)
	vol1=((Train[1]-mu1)**2)
	vol1=vol1.sum(axis=0)/(Train[1][:,1].size)
	covd1=np.diag(np.diag((vol1)))
	return covd0,covd1

	#for i in range(X_train.size):
	#	(Train[i]-mu[i])**2
"""
def	PGC2CD(Train,Classes,Valid):#calculating mean covariance and priors
	mu0=np.mean(Train[0], axis=0)
	mu1=np.mean(Train[1], axis=0)
	covd0 = []
	covd1 = []
	for n in range(len(Train[0])):
		covd0=np.append(covd0,((Train[0][n,:]-mu0)@(Train[0][n,:]-mu0).T))
	for n in range(len(Train[0])):
		covd1=np.append(covd1,((Train[1][n,:]-mu1)@(Train[1][n,:]-mu1).T))
	covd0 = np.diag(covd0)
	print(covd0)
	covd1 = np.diag(covd1)
	print(covd1)
	return covd0,covd1
	"""
def	PGC5C(Train,Classes):
	mu0=np.mean(Train[0], axis=0)
	mu1=np.mean(Train[1], axis=0)
	mu2=np.mean(Train[2], axis=0)
	mu3=np.mean(Train[3], axis=0)
	mu4=np.mean(Train[4], axis=0)
	cov0 = np.cov(Train[0].T)
	cov1 = np.cov(Train[1].T)
	cov2 = np.cov(Train[2].T)
	cov3 = np.cov(Train[3].T)
	cov4 = np.cov(Train[4].T)
	pC0 = Train[0].shape[0]/(Train[0].shape[0] + Train[1].shape[0]+Train[2].shape[0]+Train[3].shape[0]+Train[4].shape[0])
	pC1 = Train[1].shape[0]/(Train[0].shape[0] + Train[1].shape[0]+Train[2].shape[0]+Train[3].shape[0]+Train[4].shape[0])
	pC2 = Train[2].shape[0]/(Train[0].shape[0] + Train[1].shape[0]+Train[2].shape[0]+Train[3].shape[0]+Train[4].shape[0])
	pC3 = Train[3].shape[0]/(Train[0].shape[0] + Train[1].shape[0]+Train[2].shape[0]+Train[3].shape[0]+Train[4].shape[0])
	pC4 = Train[4].shape[0]/(Train[0].shape[0] + Train[1].shape[0]+Train[2].shape[0]+Train[3].shape[0]+Train[4].shape[0])

	return mu0,mu1,mu2,mu3,mu4,cov0,cov1,cov2,cov3,cov4,pC0,pC1,pC2,pC3,pC4
def	PGC5CD(Train,mu0,mu1,mu2,mu3,mu4):
	vol0=((Train[0]-mu0)**2)
	vol0=vol0.sum(axis=0)/(Train[0][:,1].size)
	covd0=np.diag(vol0)
	vol1=((Train[1]-mu1)**2)
	vol1=vol1.sum(axis=0)/(Train[1][:,1].size)
	covd1=np.diag(vol1)
	vol2=((Train[2]-mu2)**2)
	vol2=vol2.sum(axis=0)/(Train[2][:,1].size)
	covd2=np.diag(vol2)
	vol3=((Train[3]-mu3)**2)
	vol3=vol3.sum(axis=0)/(Train[3][:,1].size)
	covd3=np.diag(vol3)
	vol4=((Train[4]-mu4)**2)
	vol4=vol4.sum(axis=0)/(Train[4][:,1].size)
	covd4=np.diag(vol4)
	return covd0,covd1,covd2,covd3,covd4
	"""for z in range(Classes.size):
		mean=np.mean(Train[z], axis=0)
		mu= np.vstack((mu,mean))
	mu=np.delete(mu,0, 0)
	cov= np.empty((Train[1][1,:].size), int)
	y=[]
	for z in range(Classes.size):
		cv=np.cov(Train[z].T)
		y1 = multivariate_normal.pdf(Valid, mean=mu[z] ,cov=cv)
		y= np.append(y,[y1])
	print(y)
	wait=input("")
	return 1"""
	






        

""" ============  Generate Training and validation Data =================== """

""" Here is an example for 2D DataSet, you can change it for 7D and HS 
    Also, you can change the random_state to get different validation data """

# Here you can change your data set from 2D,7D and HS
Train = Train_2D
labels = labels_2D
Classes = np.sort(np.unique(labels))
Test=Test_2D


# Here you can change M to get different validation data
M = 65 #i changed M to very big value
X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size = 0.33, random_state = M)
X_train_class = []
for j in range(Classes.shape[0]):
    jth_class = X_train[label_train == Classes[j],:]
    X_train_class.append(jth_class)
#Visualization of first two dimension of your dataSet

for j in range(Classes.shape[0]):
    plt.scatter(X_train_class[j][:,0],X_train_class[j][:,1])
plt.show()


""" ========================  Train the Classifier ======================== """

""" Here you can train your classifier with your training data """
#PGC
if (Classes.size==2):
	mu0,mu1,cov0,cov1,pC0,pC1=PGC2C(X_train_class,Classes)
	covd0,covd1=PGC2CD(X_train_class,mu0,mu1)
else:
	mu0,mu1,mu2,mu3,mu4,cov0,cov1,cov2,cov3,cov4,pC0,pC1,pC2,pC3,pC4=PGC5C(X_train_class,Classes)
	covd0,covd1,covd2,covd3,covd4=PGC5CD(X_train_class,mu0,mu1,mu2,mu3,mu4)
#PGCDiag

#KNN
#the commented out block is using the knn library
"""
k = 5
classifiers = []
classifiers.append(neighbors.KNeighborsClassifier(k, weights='distance'))

names = [ 'K-NN_Weighted']
for name, knn in zip(names, classifiers):
    print('name: ',name)
    knn.fit(X_train, label_train)
    predictions_KNN = knn.predict(X_valid)
"""
#code-from-scratch KNN using teh euclidean distance
k=5
predictions_KNN=[]
for p in range (X_valid[:,0].size):
	neig=np.empty([1,2])
	glabelknn=[]
	for h in range (X_train[:,0].size):
		dist= (X_valid[p]-X_train[h])**2
		neig=np.vstack((neig,[[np.sqrt(np.sum(dist)),label_train[h]]]))
	neig = np.delete(neig, 0, axis=0)
	for g in range (k):
		pos = np.unravel_index(np.argmin(neig[:,0], axis=0), neig[:,0].shape)
		print(pos)
		print(neig[pos,1])
		glabelknn=np.append(glabelknn,neig[pos,1])
		neig = np.delete(neig, pos, axis=0)

	for number,count in Counter(glabelknn).most_common(1):
		value=number
	predictions_KNN.append(value)


""" ======================== Cross Validation ============================= """


""" Here you should test your parameters with validation data """
#For PCG specifically

#generate matrices
classpred=np.empty([1,1])
classvalidlabel=np.empty([1,1])
#append proper matrices for comparison which involve the posteriors for comparison
if (Classes.size==2):
	for l in range (len(X_valid)):
		y0 = multivariate_normal.pdf(X_valid[l], mean=mu0, cov=cov0)
		y1 = multivariate_normal.pdf(X_valid[l], mean=mu1, cov=cov1)
		p0=(y0*pC0)/(y0*pC0+y1*pC1)
		p1=(y1*pC1)/(y0*pC0+y1*pC1)
		if p0>p1:
			classpred=np.append(classpred,p0)
			classvalidlabel=np.append(classvalidlabel,0)
		else:
			classpred=np.append(classpred,p1)
			classvalidlabel=np.append(classvalidlabel,1)
	#For PCGDiag specifically
	classvalidlabeldiag=[]
	for l in range (len(X_valid)):
		yd0 = multivariate_normal.pdf(X_valid[l], mean=mu0, cov=covd0)
		yd1 = multivariate_normal.pdf(X_valid[l], mean=mu1, cov=covd1)
		pd0=(yd0*pC0)/(yd0*pC0+yd1*pC1)
		pd1=(yd1*pC1)/(yd0*pC0+yd1*pC1)
		if pd0>pd1:
			classvalidlabeldiag=np.append(classvalidlabeldiag,0)
		else:
			classvalidlabeldiag=np.append(classvalidlabeldiag,1)
else:#this is for 5 classes under PGC
	for l in range (len(X_valid)):
		y0 = multivariate_normal.pdf(X_valid[l], mean=mu0, cov=cov0,allow_singular=True)
		y1 = multivariate_normal.pdf(X_valid[l], mean=mu1, cov=cov1,allow_singular=True)
		y2 = multivariate_normal.pdf(X_valid[l], mean=mu2, cov=cov2,allow_singular=True)
		y3 = multivariate_normal.pdf(X_valid[l], mean=mu3, cov=cov3,allow_singular=True)
		y4 = multivariate_normal.pdf(X_valid[l], mean=mu4, cov=cov4,allow_singular=True)
		p0=(y0*pC0)/(y0*pC0+y1*pC1+y2*pC2+y3*pC3+y4*pC4)
		p1=(y1*pC1)/(y0*pC0+y1*pC1+y2*pC2+y3*pC3+y4*pC4)
		p2=(y2*pC2)/(y0*pC0+y1*pC1+y2*pC2+y3*pC3+y4*pC4)
		p3=(y3*pC3)/(y0*pC0+y1*pC1+y2*pC2+y3*pC3+y4*pC4)
		p4=(y4*pC4)/(y0*pC0+y1*pC1+y2*pC2+y3*pC3+y4*pC4)
		a=[p0,p1,p2,p3,p4]
		maxpos = a.index(max(a))
		if maxpos==0:
			classvalidlabel=np.append(classvalidlabel,1)
		elif maxpos==1:
			classvalidlabel=np.append(classvalidlabel,2)
		elif maxpos==2:
			classvalidlabel=np.append(classvalidlabel,3)
		elif maxpos==3:
			classvalidlabel=np.append(classvalidlabel,4)
		elif maxpos==4:
			classvalidlabel=np.append(classvalidlabel,5)
	#For PCGDiag specifically
	classvalidlabeldiag=[]
	for l in range (len(X_valid)):
		yd0 = multivariate_normal.pdf(X_valid[l], mean=mu0, cov=covd0)
		yd1 = multivariate_normal.pdf(X_valid[l], mean=mu1, cov=covd1)
		yd2 = multivariate_normal.pdf(X_valid[l], mean=mu2, cov=covd2)
		yd3 = multivariate_normal.pdf(X_valid[l], mean=mu3, cov=covd3)
		yd4 = multivariate_normal.pdf(X_valid[l], mean=mu4, cov=covd4)
		pd0=(yd0*pC0)/(yd0*pC0+yd1*pC1+yd2*pC2+yd3*pC3+yd4*pC4)
		pd1=(yd1*pC1)/(yd0*pC0+yd1*pC1+yd2*pC2+yd3*pC3+yd4*pC4)
		pd2=(yd2*pC2)/(yd0*pC0+yd1*pC1+yd2*pC2+yd3*pC3+yd4*pC4)
		pd3=(yd3*pC3)/(yd0*pC0+yd1*pC1+yd2*pC2+yd3*pC3+yd4*pC4)
		pd4=(yd4*pC4)/(yd0*pC0+yd1*pC1+yd2*pC2+yd3*pC3+yd4*pC4)
		a=[pd0,pd1,pd2,pd3,pd4]
		maxpos = a.index(max(a))
		if maxpos==0:
			classvalidlabeldiag=np.append(classvalidlabeldiag,1)
		elif maxpos==1:
			classvalidlabeldiag=np.append(classvalidlabeldiag,2)
		elif maxpos==2:
			classvalidlabeldiag=np.append(classvalidlabeldiag,3)
		elif maxpos==3:
			classvalidlabeldiag=np.append(classvalidlabeldiag,4)
		elif maxpos==4:
			classvalidlabeldiag=np.append(classvalidlabeldiag,5)

classvalidlabel = np.delete(classvalidlabel, (0), axis=0)
predictions_PG=classvalidlabel
predictions_PGdiag=classvalidlabeldiag
#Visualization of first two dimension of your dataSet
# The accuracy for your validation data
accuracy_PG = accuracy_score(label_valid, predictions_PG)
print('\nThe accuracy of Probabilistic Generative classifier is: ', accuracy_PG*100, '%')
accuracy_KNN = accuracy_score(label_valid, predictions_KNN)
print('\nThe accuracy of KNN classifier is: ', accuracy_KNN*100, '%')
accuracy_PGdiag = accuracy_score(label_valid, predictions_PGdiag)
print('\nThe accuracy of Probabilistic Generative classifier(diagonal) is: ', accuracy_PGdiag*100, '%')



""" ========================  Test the Model ============================== """

""" This is where you should test the testing data with your classifier """
#PGC model 
"""
testpgc=[]
for l in range (len(Test)):
	y0 = multivariate_normal.pdf(Test[l], mean=mu0, cov=cov0)
	y1 = multivariate_normal.pdf(Test[l], mean=mu1, cov=cov1)
	p0=(y0*pC0)/(y0*pC0+y1*pC1)
	p1=(y1*pC1)/(y0*pC0+y1*pC1)
	if p0>p1:
		testpgc=np.append(testpgc,0)
	else:
		testpgc=np.append(testpgc,1)

np.savetxt('2DforTestLabels.txt', testpgc, delimiter=',')
"""
"""
#KNN model

testKNN = knn.predict(Test)
np.savetxt('HyperSpectralforTestLabels.txt', testKNN, delimiter=',')
"""

#HS
"""
#KNN model

testHS = knn.predict(Test)
np.savetxt('HyperSpectralforTestLabels.txt', testHS, delimiter=',')
"""
#Refereces:
#https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
#Akash
#Junghoon
#Gayatri

