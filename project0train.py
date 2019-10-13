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
from sklearn.externals import joblib
from collections import Counter
from sklearn.cluster import KMeans

def myknn(Train,Labels,ValidLabels,car_locations): #the knn function#
	knn = KNeighborsClassifier(n_neighbors=15) # the KNN library
	knn.fit(Train,Labels)
	pred = knn.predict(ValidTrain)
	print (accuracy_score(ValidLabels,pred))
	joblib.dump(knn, 'knn.joblib')#saved training model
	n=1
	for x in range (pred.size):
		if (pred[x]==1):
			print("I found car at")
			print(ValidLocations[x])
			car_locations=np.vstack((car_locations,ValidLocations[x]))
	car_locations = np.delete(car_locations, (0), axis=0)
	#i created data clusters
	refx=car_locations[0,0]
	refy=car_locations[0,1]
	#this algorithm tries to find the optimum value of n (number of clusters)
	for l in range (0,car_locations[:,0].size-1):
		refx=car_locations[l,0]
		refy=car_locations[l,1]
		if (abs(car_locations[l+1:0]-refx)>25) or (abs(car_locations[l+1,1]-refy)>25):
			n=n+1
	print(n)
	kmeans_model = KMeans(n_clusters=n).fit(car_locations)
	centers = np.array(kmeans_model.cluster_centers_)
	#Cluster centroids calculated
	return centers
	
#load data
data = np.load('data_train.npy')
test=np.load('data_test.npy')
truth=np.load('ground_truth.npy')
valid=data
Train= np.loadtxt('traindatalow.txt')
Labels = np.loadtxt('trainlabelslow.txt')
ValidTrain=np.loadtxt('validatedata.txt')
ValidLabels=np.loadtxt('validatelabels.txt')
ValidLocations=np.loadtxt('validatelocations.txt')
imgplot = plt.imshow(valid[1900:2100,3500:4200])#selected data
plt.show()

#you can enable this chunk of code to do some K-value analysis where it allows you to select the best K.
# This is not my work but it's a good algorithm to find a usable value of "K" (The number of neighbors)

"""
error=[]
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(Train, Labels)
    pred_i = knn.predict(ValidTrain)
    error.append(np.mean(pred_i != ValidLabels))
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')
plt.show()
"""
car_locations=np.zeros(2)
centers=myknn(Train,Labels,ValidLabels,car_locations)


print(centers)

#pointing the centroids calculated
imgplot = plt.imshow(valid[1900:2100,3500:4200])

plt.plot(centers[:,0]-3500,centers[:,1]-1900,'ro')
plt.show()
np.savetxt('Traincarlocation.txt', np.round(centers,0), delimiter=',')