#!/usr/bin/python

import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt


def Allocation_2_Y(allocation):
	
	N = np.size(allocation)
	unique_elements = np.unique(allocation)
	num_of_classes = len(unique_elements)
	class_ids = np.arange(num_of_classes)

	i = 0
	Y = np.zeros(num_of_classes)
	for m in allocation:
		class_label = np.where(unique_elements == m)[0]
		a_row = np.zeros(num_of_classes)
		a_row[class_label] = 1
		Y = np.hstack((Y, a_row))

	Y = np.reshape(Y, (N+1,num_of_classes))
	Y = np.delete(Y, 0, 0)

	return Y

def HSIC_rbf(X, Y, sigma, X_kernel='Gauss', Y_kernel='Gauss', return_percent_diff=False):
	n = X.shape[0]

	gamma = 1.0/(2*sigma*sigma)

	# Create Kernel
	if (X_kernel=='Gauss'): 
		xK = sklearn.metrics.pairwise.rbf_kernel(X, gamma=gamma)
		xD = np.diag(1/np.sqrt(np.sum(xK,axis=1))) # 1/sqrt(D)
		xK = xD.dot(xK).dot(xD)						# centered Laplacian
	elif (X_kernel=='linear'): xK = X.dot(X.T)
	if (Y_kernel=='Gauss'): 
		yK = sklearn.metrics.pairwise.rbf_kernel(Y, gamma=gamma)
		yD = np.diag(1/np.sqrt(np.sum(yK,axis=1))) # 1/sqrt(D)
		yK = yD.dot(yK).dot(yD)
	elif (Y_kernel=='linear'): yK = Y.dot(Y.T)


	H = np.eye(n) - (1.0/n)*np.ones((n,n))

	## normalized 
	#C = 1.0/((n-1)*(n-1))
	#HSIC = C*np.sum((xK.dot(H)).T*yK.dot(H))

	# not normalized
	HSIC = np.sum((xK.dot(H)).T*yK.dot(H))

	if return_percent_diff:
		random_HSIC = np.array([])
		for m in range(10):
			random_allocation = np.floor(Y.shape[1]*np.random.random(X.shape[0]))
			random_Y = Allocation_2_Y(random_allocation)
			H = HSIC_rbf(X , random_Y, sigma)
			random_HSIC = np.hstack((random_HSIC,H))
	
		mean_RHSIC = np.mean(random_HSIC)
		percent_diff = (HSIC - mean_RHSIC)/HSIC
		return [HSIC, percent_diff]


	return HSIC


if __name__ == "__main__":
	hsic_list = np.array([])
	for m in range(30):
		X = np.random.random((200,2))
		Y = np.random.random((200,2))
		hsic = HSIC_rbf(X, Y, 1)
		hsic_list = np.hstack((hsic_list, hsic))
	
	for m in range(30):
		X = np.random.random((200,2))
		hsic = HSIC_rbf(X, X, 1)
		hsic_list = np.hstack((hsic_list, hsic))
	 
	#objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
	#y_pos = np.arange(len(objects))
	#performance = [10,8,6,4,2,1]
	 
	y_pos = np.arange(len(hsic_list))
	plt.bar(y_pos, hsic_list, align='center', alpha=0.5)
	#plt.xticks(y_pos, objects)
	plt.xlabel('iterations of random HSIC')
	plt.ylabel('HSIC')
	plt.title('Low vs High HSIC')
	 
	plt.show()
	
	import pdb; pdb.set_trace()

