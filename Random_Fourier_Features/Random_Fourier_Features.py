#!/usr/bin/env python
#Paper : https://drive.google.com/file/d/0B2GQI7-ZH4djdm85SGg4dUhBVVE/view


import numpy as np
import sklearn.metrics
import numpy.matlib
from sklearn.preprocessing import normalize
from numpy import genfromtxt
import time

import sklearn.metrics
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier



#	K(x, y) = exp(-gamma ||x-y||^2)
#	sigma = sqrt( 1/(2*gamma) )
#	gamma = 1/(2*sigma^2)

np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)

def RFF(X, nrmlize, m, sigma):
	N = X.shape[0]
	d = X.shape[1]

	phase_shift = 2*np.pi*np.random.rand(1, m)
	phase_shift = np.matlib.repmat(phase_shift, N, 1)
	rand_proj = np.random.randn(d, m)/(sigma)

	P = np.cos(X.dot(rand_proj) + phase_shift)
	K = (2.0/m)*P.dot(P.T)

	return K



if __name__ == "__main__":
	#X = np.array([[0,0],[0,1],[0,-1],[4,4],[4,5],[3,4],[4,3]], dtype='f')	 
	#X = genfromtxt('../dataset/data_4.csv', delimiter=',')
	#X = genfromtxt('../dataset/breast-cancer.csv', delimiter=',')
	#X = genfromtxt('../dataset/facial_85.csv', delimiter=',')
	#X = genfromtxt('../dataset/moon_30000x4.csv', delimiter=',')
	
	
	sigma = 0.5
	m = 2000


	start_time = time.time() 
	K = RFF(X, True, m, sigma)
	K_time = (time.time() - start_time)


	gamma = 1.0/(2*sigma*sigma)
	start_time = time.time() 
	rbk = sklearn.metrics.pairwise.rbf_kernel(X, gamma=gamma)
	rbk_time = (time.time() - start_time)


	start_time = time.time() 
	rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=2000)
	Z = rbf_feature.fit_transform(X)
	rff_K = Z.dot(Z.T)
	rff_time = (time.time() - start_time)


	print K_time, rbk_time, rff_time

	import pdb; pdb.set_trace()

