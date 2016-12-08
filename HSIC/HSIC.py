#!/usr/bin/python

import sklearn.metrics
import numpy as np

def HSIC_rbf(X, Y, sigma):
	n = X.shape[0]

	gamma = 1.0/(2*sigma*sigma)
	xK = sklearn.metrics.pairwise.rbf_kernel(X, gamma=gamma)
	yK = sklearn.metrics.pairwise.rbf_kernel(Y, gamma=gamma)
	H = np.eye(n) - (1.0/n)*np.ones((n,n))
	C = 1.0/((n-1)*(n-1))

	HSIC = C*np.sum((xK.dot(H)).T*yK.dot(H))
	print HSIC

	HSIC = C*np.trace(xK.dot(H).dot(yK).dot(H))
	print HSIC

	import pdb; pdb.set_trace()




#X = np.array([[1,2],[2,3],[1,1],[1,3]], dtype='f')	 #	rows are samples
X = np.random.random((200,2))
Y = np.random.random((200,2))

HSIC_rbf(X, X, 1)

#	[[ 1.          0.36787945  0.60653067  0.60653067]
#	 [ 0.36787945  1.          0.082085    0.60653067]
#	 [ 0.60653067  0.082085    1.          0.13533528]
#	 [ 0.60653067  0.60653067  0.13533528  1.        ]]
