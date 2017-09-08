#!/usr/bin/env python
#	Assume symmetric matrix

import numpy as np
import time 



np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)

class CPM():
	#	A is a symmetric np array	::::  VERY IMPORTANT :::::::
	#	Num_of_eigV : return the number of eigen vectors
	#	style : 'dominant_first' , 'least_dominant_first', 'largest_first', 'smallest_first',  dominate applies absolute value first
	def __init__(self, A, Num_of_eigV=1, style='dominant_first', init_with_power_method=True):
		self.exit_threshold = 0.000001
		self.n = A.shape[0]	
		self.top_pcnt = int(np.round(self.n/50.0))		# number of rows to keep
		self.A = A.copy()
		self.Num_of_eigV = Num_of_eigV
		self.init_with_power_method = init_with_power_method

		if self.Num_of_eigV > self.n: self.Num_of_eigV = self.n
		if self.top_pcnt < 1: self.top_pcnt = 1

		if style == 'dominant_first': self.dominant_first(A)
		if style == 'least_dominant_first': pass
		if style == 'largest_first': pass
		if style == 'smallest_first': pass

	def dominant_first(self, A):
		self.eigValues = np.empty((1, 0))
		self.eigVect = np.empty((self.n, 0))

		
		for eig_id in range(self.Num_of_eigV):
			x = self.CPM(A)
			eigValue = np.mean(A.dot(x)/x)
			
			self.eigVect = np.hstack((self.eigVect,x))	
			self.eigValues = np.append(self.eigValues,eigValue)

			A = A - eigValue*x.dot(x.T)



	def power_method(self, A, x):
		loop = True

		while loop:
			z = A.dot(x)

			eigs = z/x
			eig_diff = np.absolute(np.max(eigs) - np.min(eigs))
			if(eig_diff < 0.001): break;
			x = z/x.T.dot(z)
		return x

	def CPM_single(self, A):												#	A is a symmetric PSD np array
		x = np.random.randn(self.n,1)
		x = x/np.linalg.norm(x)
		if self.init_with_power_method: x = self.power_method(A, x)

		z = A.dot(x)
		c = np.absolute(z - x)
		loop = True
	
	
		while loop: 
			i = np.argmax(c)
			denom = x.T.dot(z)
	
			yi = (z/(x.T.dot(z)))[i]
			xi = x[i]
			
			z = z + (A[:, i]*(yi-xi)).reshape(self.n,1)
	
			x[i] = yi
			yn = np.linalg.norm(x)
			z = z/yn
			x = x/yn
	
			c = np.absolute(x - z/denom)
			max_c = np.max(c)

			if(max_c < self.exit_threshold): break;

		return x

	def CPM(self, A):												#	A is a symmetric PSD np array
		x = np.random.randn(self.n,1)
		x = x/np.linalg.norm(x)
		if self.init_with_power_method: x = self.power_method(A, x)

		z = A.dot(x)
		c = np.absolute(z - x)
		loop = True
	
	
		while loop: 
			i = np.argpartition(c.reshape((self.n,)), -self.top_pcnt)[-self.top_pcnt:] 
			denom = x.T.dot(z)

			yi = (z/(x.T.dot(z)))[i]
			xi = x[i]

			z = z + (A[:, i].dot(yi-xi))
	
			x[i] = yi
			yn = np.linalg.norm(x)
			z = z/yn
			x = x/yn
	
			c = np.absolute(x - z/denom)
			max_c = np.max(c)

			if(max_c < 100000*self.exit_threshold): self.top_pcnt = int(self.top_pcnt/2) + 1
			if(max_c < 10000*self.exit_threshold): self.top_pcnt = int(self.top_pcnt/2) + 1
			if(max_c < 1000*self.exit_threshold): self.top_pcnt = int(self.top_pcnt/2) + 1
			if(max_c < 100*self.exit_threshold): self.top_pcnt = int(self.top_pcnt/2) + 1
			if(max_c < 10*self.exit_threshold): self.top_pcnt = int(self.top_pcnt/2) + 1
			if(max_c < 5*self.exit_threshold): self.top_pcnt = int(self.top_pcnt/2) + 1
			if(max_c < self.exit_threshold): break;

		return x


if __name__ == "__main__":
	from eig_sorted import *
	d = 2
	A = np.random.randn(100,100)
	A = A.dot(A.T)


	#	Ran coordinate wise power method
	start_time = time.time() 
	cpm = CPM(A, Num_of_eigV=d, style='dominant_first')
	cpm_time = (time.time() - start_time)


	#	Ran regular eig decomposition method
	start_time = time.time() 
	[V,D] = eig_sorted(A)
	eig_time = (time.time() - start_time)



	print cpm.eigValues , D[0:d]
	print np.hstack((cpm.eigVect[0:4, 0:d], V[0:4, 0:d]))
	print 'cpm_time : ' , cpm_time , '  ,  ' , 'eig_time : ', eig_time


	import pdb; pdb.set_trace()
