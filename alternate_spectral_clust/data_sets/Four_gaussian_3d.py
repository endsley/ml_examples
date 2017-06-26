#!/usr/bin/python


from numpy import genfromtxt
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


N = 1000
data = genfromtxt('Four_gaussian_3D.csv', delimiter=',')	
data = data[:,0:3]

clf = KMeans(n_clusters=4)
kmeans = clf.fit(data)
center_1 = kmeans.cluster_centers_[0,:]
center_2 = kmeans.cluster_centers_[1,:]
center_3 = kmeans.cluster_centers_[2,:]
center_4 = kmeans.cluster_centers_[3,:]

D1 = np.random.normal(loc = center_1, size=(N/4,3))
D2 = np.random.normal(loc = center_2, size=(N/4,3))
D3 = np.random.normal(loc = center_3, size=(N/4,3))
D4 = np.random.normal(loc = center_4, size=(N/4,3))

data = np.vstack((D1,D2,D3,D4))
noise = np.random.uniform(low=-20, high=20, size=(N,1))
data = np.hstack((data,noise))

label1 = np.vstack((np.zeros((N/4,1)),np.ones((N/4,1)),np.ones((N/4,1)),np.zeros((N/4,1))))
label2 = np.vstack((np.zeros((N/2,1)),np.ones((N/2,1))))

if False:
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(data[:,0], data[:,1], data[:,2], c='b', marker='o')
	ax.set_xlabel('Feature 1')
	ax.set_ylabel('Feature 2')
	ax.set_zlabel('Feature 3')
	ax.set_title('Original Clustering')
	plt.show()

np.savetxt('Four_gaussian_3D_' + str(N) + '.csv', data, delimiter=',', fmt='%f')


np.savetxt('Four_gaussian_3D_' + str(N) + '_original_label.csv', label1, delimiter=',', fmt='%f')
np.savetxt('Four_gaussian_3D_' + str(N) + '_alt_label.csv', label2, delimiter=',', fmt='%f')


import pdb; pdb.set_trace()
