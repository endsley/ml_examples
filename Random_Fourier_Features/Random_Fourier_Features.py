#!/usr/bin/python
#Paper : https://drive.google.com/file/d/0B2GQI7-ZH4djdm85SGg4dUhBVVE/view


import numpy as np
import sklearn.metrics
import numpy.matlib
from sklearn.preprocessing import normalize
from numpy import genfromtxt


#	K(x, y) = exp(-gamma ||x-y||^2)
#	sigma = sqrt( 1/(2*gamma) )
#	gamma = 1/(2*sigma^2)

np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)

def RFF(X, nrmlize, m, sigma):
	if nrmlize:
		X = normalize(X, norm='l2', axis=1)
	
	gamma = 1.0/(2*sigma*sigma)

	n = X.shape[0]
	d = X.shape[1]
	rbk = sklearn.metrics.pairwise.rbf_kernel(X, gamma=gamma)
	

	u = np.random.randn(d, m)/(sigma)
	b = np.random.rand(1, m)
	b = np.matlib.repmat(b, n, 1)*2*np.pi
	
	y = np.cos(X.dot(u) + b)
	K = y.dot(y.T)*2/m
	error = np.linalg.norm(rbk - K)

	print rbk[0:20,0:20]
	print K[0:20,0:20] , '\n'
	print 'Error from RFF : ' , error , '\n\n'
	import pdb; pdb.set_trace()
	return error


if __name__ == "__main__":
	X = genfromtxt('data_4.csv', delimiter=',')
	#X = np.array([[0,0],[0,1],[0,-1],[4,4],[4,5],[3,4],[4,3]], dtype='f')	 
	m = 40000
	RFF(X, True, m, 0.8)






