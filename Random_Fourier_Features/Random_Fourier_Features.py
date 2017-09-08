#!/usr/bin/env python
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



	b = 2*np.pi*np.random.rand(1, m)
	b = np.matlib.repmat(b, self.N, 1)

	self.phase_shift = torch.from_numpy(b)
	self.phase_shift = Variable(self.phase_shift.type(self.dtype), requires_grad=False)

	u = np.random.randn(self.output_d, m)/(self.sigma)
	self.rand_proj = torch.from_numpy(u)
	self.rand_proj = Variable(self.rand_proj.type(self.dtype), requires_grad=False)

	u2 = np.random.randn(self.d, m)/(self.sigma)
	self.rand_proj2 = torch.from_numpy(u2)
	self.rand_proj2 = Variable(self.rand_proj2.type(self.dtype), requires_grad=False)

	P = torch.cos(torch.mm(input_data,self.rand_proj) + self.phase_shift)

	K = torch.mm(P, P.transpose(0,1))
	K = (2.0/m)*K
	
	#K = K + 0.02
	#K = 1 - torch.sigmoid(-7*K + 3.5)
	#K = 1 - torch.sigmoid(-8*K + 5)
	#K = K.clamp(min=0)	#clamp doesn't seem to do back prop
	K = torch.abs(K)





#	if nrmlize:
#		X = normalize(X, norm='l2', axis=1)
#	
#	gamma = 1.0/(2*sigma*sigma)
#
#	n = X.shape[0]
#	d = X.shape[1]
#	rbk = sklearn.metrics.pairwise.rbf_kernel(X, gamma=gamma)
#	
#	u = np.random.randn(d, m)/(sigma)
#	b = np.random.rand(1, m)
#	b = np.matlib.repmat(b, n, 1)*2*np.pi
#	
#	y = np.cos(X.dot(u) + b)
#	K = y.dot(y.T)*2/m
#	error = np.linalg.norm(rbk - K)
#
#	print rbk[0:20,0:20]
#	print K[0:20,0:20] , '\n'
#	print 'Error from RFF : ' , error , '\n\n'
#	import pdb; pdb.set_trace()
#	return error


if __name__ == "__main__":
	#X = np.array([[0,0],[0,1],[0,-1],[4,4],[4,5],[3,4],[4,3]], dtype='f')	 
	X = genfromtxt('data_4.csv', delimiter=',')
	sigma = 0.5
	m = 2000


	RFF(X, True, m, sigma)



	gamma = 1.0/(2*sigma*sigma)
	rbk = sklearn.metrics.pairwise.rbf_kernel(X, gamma=gamma)





