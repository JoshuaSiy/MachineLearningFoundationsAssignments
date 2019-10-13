from matplotlib import pyplot as plt
import numpy as np
import mlp
from sklearn.model_selection import train_test_split
#load data
data = np.load('dataSet.npy')
X_train, X_valid, y_train, y_valid = train_test_split(data[:,0:2], np.expand_dims(data[:,2],axis=1), test_size=0.3)
#Set up Neural Network

eta=0.01
niterations=1500
data_in = X_train
target_in = y_train
hidden_layers =3
NN = mlp.mlp(X_train,y_train,hidden_layers)
NN.earlystopping(X_train,y_train,X_valid,y_valid,eta,niterations)
NN.confmat(X_train,y_train)
NN.confmat(X_valid,y_valid)
NN.confmat(data[:,0:2],np.expand_dims(data[:,2],axis=1))
    

y3=data[:,2]
x3=data[:,0:2]
idx3 = y3.argsort()[::-1]   
y3= y3[idx3]
x3 = x3[idx3]


x1plot=np.linspace(0, 5, num=10)
x1plot=np.linspace(0, 5, num=10)
for x in range(hidden_layers):
    plt.plot(x1plot, (x1plot*NN.weights1[0,x]-NN.weights1[2,x])/-NN.weights1[1,x], label=("w"+str(x)))

plt.xlim(0, 5)
plt.ylim(10, -10)
plt.legend()


c1=0
c2=0
for i in range(idx3.size):
	if (y3[i]==1):
		plt.scatter(x3[i,0],x3[i,1],c='m',label=1 if c2==0 else "", marker='s')
		c2=c2+1
	elif (y3[i]==0):
		plt.scatter(x3[i,0],x3[i,1],c='c',label=0 if c1==0 else "", marker='s')
		c1=c1+1

plt.show()

#Analyze Neural Network Performance
