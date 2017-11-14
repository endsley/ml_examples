#!/usr/bin/env python

import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

def HSIC_rbf(X, Y, sigma):
	if (X.shape[0] != Y.shape[0]): 
		print 'Error : size of X and Y must be equal'
		return

	n = X.shape[0]
	if len(X.shape) == 1: X = X.reshape((n,1))
	if len(Y.shape) == 1: Y = Y.reshape((n,1))


	gamma = 1.0/(2*sigma*sigma)

	xK = sklearn.metrics.pairwise.rbf_kernel(X, gamma=gamma)
	yK = sklearn.metrics.pairwise.rbf_kernel(Y, gamma=gamma)
	#yK = Y.dot(Y.T)

	H = np.eye(n) - (1.0/n)*np.ones((n,n))
	C = 1.0/((n-1)*(n-1))

	HSIC = C*np.sum((xK.dot(H)).T*yK.dot(H))
	#HSIC = np.sum((xK.dot(H)).T*yK.dot(H))

	#print HSIC
	#HSIC = C*np.trace(xK.dot(H).dot(yK).dot(H))
	#print HSIC

	return HSIC


if __name__ == "__main__":
	hsic_list = np.array([])
#	for m in range(30):
#		X = np.random.random((200,2))
#		Y = np.random.random((200,2))
#		hsic = HSIC_rbf(X, Y, 1)
#		hsic_list = np.hstack((hsic_list, hsic))
#	
#	for m in range(30):
#		X = np.random.random((200,2))
#		hsic = HSIC_rbf(X, X, 1)
#		hsic_list = np.hstack((hsic_list, hsic))
	 
	for m in range(30):
		X = np.random.random((200,2))
		X[X>=0.5] = 1
		X[X<0.5] = 0

		hsic = HSIC_rbf(X, X, 1)
		hsic_list = np.hstack((hsic_list, hsic))

	angle = 45	#np.pi/2
	rot = np.array([[ np.cos(angle), -np.sin(angle)],[ np.sin(angle), np.cos(angle)]])
	for m in range(30):
		X = np.random.random((200,2))
		X[X>=0.5] = 1
		X[X<0.5] = 0

		
		Y = abs(X - 1)
		Y = Y.dot(rot)
		#Y = np.random.random((200,2))

		hsic = HSIC_rbf(X, Y, 1)
		hsic_list = np.hstack((hsic_list, hsic))
		#import pdb; pdb.set_trace()

	y_pos = np.arange(len(hsic_list))
	plt.bar(y_pos, hsic_list, align='center', alpha=0.5)
	#plt.xticks(y_pos, objects)
	plt.xlabel('iterations of random HSIC')
	plt.ylabel('HSIC')
	plt.title('Low vs High HSIC')
	 
	plt.show()
	
	import pdb; pdb.set_trace()

