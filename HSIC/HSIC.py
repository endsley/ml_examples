#!/usr/bin/python

import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

def HSIC_rbf(X, Y, sigma):
	n = X.shape[0]

	gamma = 1.0/(2*sigma*sigma)
	xK = sklearn.metrics.pairwise.rbf_kernel(X, gamma=gamma)
	yK = sklearn.metrics.pairwise.rbf_kernel(Y, gamma=gamma)
	H = np.eye(n) - (1.0/n)*np.ones((n,n))
	C = 1.0/((n-1)*(n-1))

	HSIC = C*np.sum((xK.dot(H)).T*yK.dot(H))

	#print HSIC
	#HSIC = C*np.trace(xK.dot(H).dot(yK).dot(H))
	#print HSIC

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

