#!/usr/bin/python

import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from sklearn import preprocessing

def Allocation_2_Y(allocation):
	
	N = np.size(allocation)
	unique_elements = np.unique(allocation)
	num_of_classes = len(unique_elements)
	class_ids = np.arange(num_of_classes)

	i = 0
	Y = np.zeros(num_of_classes)
	for m in allocation:
		class_label = np.where(unique_elements == m*np.ones(unique_elements.shape))[0]

		a_row = np.zeros(num_of_classes)
		a_row[class_label] = 1
		Y = np.hstack((Y, a_row))

	Y = np.reshape(Y, (N+1,num_of_classes))
	Y = np.delete(Y, 0, 0)

	return Y

def calc_gamma(X, gamma):
	print 'gamma = ', gamma
	if gamma != None: return gamma
	print '----------'

	#X = preprocessing.scale(X)
	d_matrix = sklearn.metrics.pairwise.pairwise_distances(X, metric='euclidean')
	sigma = np.median(d_matrix)
	if(sigma == 0): sigma = np.mean(d_matrix)
	gamma = 1.0/(2*sigma*sigma)
	return gamma



def HSIC(X, Y, X_kernel='Gaussian', Y_kernel='Delta', gamma=None): # X is n by d, n=# of sample, d=# of features
	if (X.shape[0] != Y.shape[0]): 
		print 'Error : size of X and Y must be equal'
		exit()
	
	n = X.shape[0]
	if len(X.shape) == 1: X = X.reshape((n,1))
	if len(Y.shape) == 1: Y = Y.reshape((n,1))


	#import pdb; pdb.set_trace()	
	if X_kernel == 'Gaussian':
		xK = sklearn.metrics.pairwise.rbf_kernel(X, gamma=calc_gamma(X, gamma))

	elif X_kernel == 'Linear':
		xK = X.dot(X.T)
	elif X_kernel == 'Delta':
		if X.shape[1] > 1:
			print 'Error : Cannot use delta kernel for multiple column Y'
			exit()

		X = Allocation_2_Y(X)
		xK = X.dot(X.T)
		

	if Y_kernel == 'Gaussian':
		#Y = preprocessing.scale(Y)

		d_matrix = sklearn.metrics.pairwise.pairwise_distances(Y, metric='euclidean')
		sigma = np.median(d_matrix)
		if(sigma == 0): sigma = np.mean(d_matrix)
		gamma = 1.0/(2*sigma*sigma)
		gamma = 1

		yK = sklearn.metrics.pairwise.rbf_kernel(Y, gamma=calc_gamma(Y, gamma))
	elif Y_kernel == 'Linear':
		yK = Y.dot(Y.T)
	elif Y_kernel == 'Delta':
		if Y.shape[1] > 1:
			print 'Error : Cannot use delta kernel for multiple column Y'
			exit()
			
		Y = Allocation_2_Y(Y)
		yK = Y.dot(Y.T)

	gamma = 1.0/2
	n = X.shape[0]
	xK = sklearn.metrics.pairwise.rbf_kernel(X, gamma=gamma)
	yK = sklearn.metrics.pairwise.rbf_kernel(Y, gamma=gamma)

	H = np.eye(n) - (1.0/n)*np.ones((n,n))
	C = 1.0/((n-1)*(n-1))

	#HSIC_value = C*np.trace(H.dot(xK).dot(H).dot(Y).dot(Y.T))
	HSIC_value = C*np.sum((xK.dot(H)).T*yK.dot(H))
	return HSIC_value





if __name__ == "__main__":

	#	Test 1
	hsic_list = np.array([])
	for m in range(30):
		X = np.random.random((200,2))
		Y = np.random.random((200,2))
		#hsic = HSIC_rbf(X, Y)

		hsic = HSIC(X, Y, X_kernel='Gaussian', Y_kernel='Linear')

		hsic_list = np.hstack((hsic_list, hsic))
	
	for m in range(30):
		X = np.random.random((200,2))
		#hsic = HSIC_rbf(X, X)
		hsic = HSIC(X, X, X_kernel='Gaussian', Y_kernel='Linear')
		hsic_list = np.hstack((hsic_list, hsic))
	 
	plt.subplot(211)
	y_pos = np.arange(len(hsic_list))
	plt.bar(y_pos, hsic_list, align='center', alpha=0.5)
	#plt.xticks(y_pos, objects)
	plt.xlabel('iterations of random HSIC')
	plt.ylabel('HSIC')
	plt.title('Low vs High HSIC X=Gaussian, Y=Linear')
	 

	#	Test 2
	hsic_list = np.array([])
	for m in range(30):
		X = np.random.random((200,2))
		Y = np.random.random((200,2))
		#hsic = HSIC_rbf(X, Y)

		hsic = HSIC(X, Y, X_kernel='Gaussian', Y_kernel='Gaussian')

		hsic_list = np.hstack((hsic_list, hsic))
	
	for m in range(30):
		X = np.random.random((200,2))
		#hsic = HSIC_rbf(X, X)
		hsic = HSIC(X, X, X_kernel='Gaussian', Y_kernel='Gaussian')
		hsic_list = np.hstack((hsic_list, hsic))
	 
	plt.subplot(212)
	y_pos = np.arange(len(hsic_list))
	plt.bar(y_pos, hsic_list, align='center', alpha=0.5)
	#plt.xticks(y_pos, objects)
	plt.xlabel('iterations of random HSIC')
	plt.ylabel('HSIC')
	plt.title('Low vs High HSIC X=Gaussian, Y=Gaussian')
	
	plt.show()
	
	import pdb; pdb.set_trace()


