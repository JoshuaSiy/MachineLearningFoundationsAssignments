# -*- coding: utf-8 -*-
"""
File:   hw01.py
Author: Joshua Siy	
Date:   started 6/9/2018
Desc:   I tried my best, Dr. Zare or Mr.Wells,or Mr.McCurley, or Ms. Guo.
    
"""

""" =======================  Import dependencies ========================== """

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

plt.close('all') #close any open plots

"""
===============================================================================
===============================================================================
============================ Question 1 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Function definitions ========================== """

def generateUniformData(N, l, u, gVar):
	'''generateUniformData(N, l, u, gVar): Generate N uniformly spaced data points 
    in the range [l,u) with zero-mean Gaussian random noise with variance gVar'''
	# x = np.random.uniform(l,u,N)
	step = (u-l)/(N);
	x = np.arange(l+step/2,u+step/2,step)
	e = np.random.normal(0,gVar,N)
	t = np.sinc(x) + e
	return x,t

def plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'bo') #plot training data
    p2 = plt.plot(x2, t2, 'g') #plot true value
    if(x3 is not None):
        p3 = plt.plot(x3, t3, 'r') #plot training data

    #add title, legend and axes labels
    plt.ylabel('t') #label x and y axes
    plt.xlabel('x')
    
    if(x3 is None):
        plt.legend((p1[0],p2[0]),legend)
    else:
        plt.legend((p1[0],p2[0],p3[0]),legend)
        
    plt.show()
def plotDataTest(x1,t1,x2,t2,x_test,t_test,x4,t4,x3=None,t3=None,legend=[]):
	'''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
	training data, the true function, and the estimated function'''
	p1 = plt.plot(x1, t1, 'bo') #plot training data
	p2 = plt.plot(x2, t2, 'g') #plot true value
	p_test = plt.plot(x_test, t_test,'yo') #plot test data
	p_testcurve= plt.plot(x4, t4,'m')#plot test curve
	if(x3 is not None):
		p3 = plt.plot(x3, t3, 'r') #plot training data
    #add title, legend and axes labels
		plt.ylabel('t') #label x and y axes
		plt.xlabel('x')
	if(x3 is None):
		plt.legend((p1[0],p2[0],p_test[0]),legend)
	else:
		plt.legend((p1[0],p2[0],p_test[0],p_testcurve[0],p3[0]),legend)
        
	plt.show()
def plotDataError(x1,e1,x2,e2,legend=[]):#plot error 
	'''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
	training data, the true function, and the estimated function'''
	p1 = plt.plot(x1, e1, 'b') #plot errors of train
	p2 = plt.plot(x2, e2, 'r') #plot errors of test
	plt.ylabel('ERMS') #label x and y axes
	plt.xlabel('M')
	plt.legend((p1[0],p2[0],),legend)
	plt.show()

def plotconverge(like,post,range,legend=[]):
	p1 = plt.plot(range, like, 'r') #plot likelihood estimation
	p2 = plt.plot(range, post, 'g') #plot posterior
	plt.legend((p1[0],p2[0],),legend)
    #add title, legend and axes label
	plt.ylabel('data') #label x and y axes
	plt.xlabel('range')
	plt.legend((p1[0],p2[0]),legend)
	plt.show()
		

"""
This seems like a good place to write a function to learn your regression
weights!
    
"""
def fitdata(x,t,M):
	X = np.array([x**m for m in range(M+1)]).T
	w = np.linalg.inv(X.T@X)@X.T@t
	return w

	
""" ======================  Variable Declaration ========================== """

l = 0 #lower bound on x
u = 10 #upper bound on x
N = 50 #number of samples to generate
gVar = .25 #variance of error distribution
M = 19 #regression model order
""" =======================  Generate Training Data ======================= """
data_uniform  = np.array(generateUniformData(N, l, u, gVar)).T

x1 = data_uniform[:,0]
t1 = data_uniform[:,1]

x2 = np.arange(l,u,0.001)  #get equally spaced points in the xrange
t2 = np.sinc(x2) #compute the true function value
 
""" ========================  Train the Model ============================= """

w = fitdata(x1,t1,M)
x3 = np.arange(l,u,0.001)  #get equally spaced points in the xrange
X = np.array([x3**m for m in range(w.size)]).T
print(X)
t3= X@w #compute the predicted value

#plotData(x1,t1,x2,t2,x3,t3,['Training Data', 'True Function', 'Estimated\nPolynomial'])
""" ======================== Generate Test Data =========================== """


"""This is where you should generate a validation testing data set.  This 
should be generated with different parameters than the training data!   """
   

data_uniform_test= np.array(generateUniformData(30, l, u,gVar )).T#generating own test data

x_test= data_uniform_test[:,0]#extract x value of test data
t_test= data_uniform_test[:,1]#extract results

#this block until plotDataTest is actually unnecessary
x4 = np.arange(l,u,0.001)  #get equally spaced points in the xrange
XT = np.array([x4**m for m in range(w.size)]).T
t4= XT@w #compute the predicted value

#plotDataTest(x1,t1,x2,t2,x_test,t_test,x4,t4,x3,t3,['Training Data', 'True Function','Test Data','Test Polynomial','Estimated\nPolynomial'])
#plt.close()

Mdivided=np.arange(0,M,1)#create labels for M to be used in plotting
print(Mdivided)
testerms=[]#create blank list that will be appended laters for test ERMS values
trainerms=[]#create blank list that will be appended laters for train ERMS values
for x in range(M):# this block of code tries to determine the error of the values as the model rises from 0 to M-1.
	w = fitdata(x1,t1,x)#fitting data as M changes
	X = np.array([x1**m for m in range(w.size)]).T#scale and replicate x values of train with respect to weight size
	XT = np.array([x_test**m for m in range(w.size)]).T#scale and replicate x values of test with respect to weight size
	ttrainpred= X@w#predicted values for train
	ttestpred=XT@w#predicted values for test
	trainpredandobs=(ttrainpred-t1)@(ttrainpred-t1).T#a block to be used for computing ERMS for train
	testpredandobs=(ttestpred-t_test)@(ttestpred-t_test).T#a block to be used for computing test ERMS
	trainerms.append(np.sqrt(trainpredandobs/trainpredandobs.size))#appending values in trainermslist squared after being divided by sample size made
	testerms.append(np.sqrt(testpredandobs/testpredandobs.size))#appending values in testermslist squared after being divided by sample size made
#plotDataError(Mdivided,trainerms,Mdivided,testerms,['Train Error','Test Error'])#plotting it
#plt.close()
"""
Discussing which Model value to use for the first problem. I believe due to the error generated, it is sort of hard to tell which M since the errors tend to be flunctuating

So given whatever Error diference there is between the training and the testing data. We must select the models that doesn't produce so much error against each other.

So i believe Model sizes of 0-6 is still acceptable for the methods because the errors aren't very big compared to going beyond. However some tests are made observing errors get bigger and bigger depending on the sample size. 

Amount of data also affects the data that it can overfit or underfit
"""
trueMu = 4
trueVar = 2

#Initial prior distribution mean and variance (You should change these parameters to see how they affect the ML and MAP solutions)
priorMu = 8
priorVar =0.2

numDraws = 200 #Number of draws from the true distribution

"""
===============================================================================
===============================================================================
============================ Question 2 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Variable Declaration ========================== """
"""========================== Plot the true distribution =================="""
#plot true Gaussian function
step = 0.01
l = -20
u = 20
x = np.arange(l+step/2,u+step/2,step)
plt.figure(0)
p1 = plt.plot(x, norm(trueMu,trueVar).pdf(x), color='b')
#plt.title('Known "True" Distribution')
#plt.show()
MLEcoll=[]#create list for MLE records
MAPcoll=[]#create lsit for MAP records
flipResult = []#create list for gaussian generations made

"""========================= Perform ML and MAP Estimates =================="""
#Calculate posterior and update prior for the given number of draws
for flip in range(numDraws):
	flipResult.append(np.random.normal(trueMu,math.sqrt(trueVar),1))#gaussianfunction
	MLE=sum(flipResult)/len(flipResult)#get average or likelihood estimation
	MLEcoll.append(MLE)#add this to the list
	MAP=(trueVar*priorMu+len(flipResult)*priorVar*MLE)/(len(flipResult)*priorVar+trueVar)#calculates mean of posterior
	MAPcoll.append(MAP)#adds this to MAP list
	print('Frequentist/Maximum Likelihood Probability of Heads:' + str(MLE))#print the value of likelihood results
	print('Bayesian/MAP Probability of Heads:' + str(MAP))#print the value of posterior results
	priorMu=MAP#replace prior with posterior
	priorVar=trueVar/(1+flip)#calcualte and replace prior variance with the posterior variance
numdrawsinnp=np.arange(0,numDraws,1)#generate an array to be used as label for plot regarding number of draws


"""
You should add some code to visualize how the ML and MAP estimates change
with varying parameters.  Maybe over time?  There are many differnt things you could do!
"""

#plotconverge(np.asarray(MLEcoll),np.asarray(MAPcoll),numdrawsinnp,['MLE','MAP'])#plot the data of MLE and MAP

# To Dr. Zare or Teaching assistants
# the values in this assignment were reset back to the original given at best ability. I hope i do not get penalized for changing the values.
#To the changes and experiments done, please refer to the PDF.
# Where i explained what experiments i have done with this code....
#TESTING for travis