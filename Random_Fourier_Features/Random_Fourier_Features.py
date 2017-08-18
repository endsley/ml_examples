#!/usr/bin/python
#Paper : https://drive.google.com/file/d/0B2GQI7-ZH4djdm85SGg4dUhBVVE/view


import numpy as np
import sklearn.metrics
import numpy.matlib
from sklearn.preprocessing import normalize

#	K(x, y) = exp(-gamma ||x-y||^2)
#	sigma = sqrt( 1/(2*gamma) )
#	gamma = 1/(2*sigma^2)

np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)

def RFF(X, nrmlize, m):
	if nrmlize:
		X = normalize(X, norm='l2', axis=1)
	
	n = X.shape[0]
	d = X.shape[1]
	rbk = sklearn.metrics.pairwise.rbf_kernel(X, gamma=0.5)
	#print rbk
	
	u = np.random.randn(d, m)
	b = np.random.rand(1, m)
	b = np.matlib.repmat(b, n, 1)*2*np.pi
	
	y = np.cos(X.dot(u) + b)
	K = y.dot(y.T)*2/m
	error = np.linalg.norm(rbk - K)
	#print K , '\n'
	#print 'Error from RFF : ' , error , '\n\n'

	return error



#	This example shows that normalized RFF can be approximated with lower variance
#	The purpose of the normalized version is NOT to estimate the original kernel
#	but a proposal of another kernel that when estimated has a lower variance.
m = 100
X = np.array([[1,2],[2,3],[1,1],[1,3]], dtype='f')	 

RFF_error = []
for p in range(1000):
	RFF_error.append(RFF(X, False, m))

print np.mean(RFF_error)

RFF_error = []
for p in range(1000):
	RFF_error.append(RFF(X, True, m))

print np.mean(RFF_error)



import pdb; pdb.set_trace()
