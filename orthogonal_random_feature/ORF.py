#!/usr/bin/env python

import numpy as np
import sys
import numpy.matlib
import sklearn.metrics
from time import perf_counter
from scipy.linalg import hadamard

class ORF():
	def __init__(self):
		pass

	def RFF(self, X, m, σ):
		N = X.shape[0]
		d = X.shape[1]
	
		phase_shift = 2*np.pi*np.random.rand(1, m)
		phase_shift = np.matlib.repmat(phase_shift, N, 1)
		rand_proj = np.random.randn(d, m)/(σ)
	
		P = np.cos(X.dot(rand_proj) + phase_shift)
		K = (2.0/m)*P.dot(P.T)
	
		return K

	def SORF(self, X, σ, α):	# α repeats the number of Wᵴ matrices
		N = X.shape[0]
		d = X.shape[1]
	
		phase_shift = 2*np.pi*np.random.rand(1, m)
		phase_shift = np.matlib.repmat(phase_shift, N, 1)
		Wᵴ = self.get_Wᵴ(X.T, σ, α)
	
		P = np.cos(X.dot(Wᵴ.T) + phase_shift)
		K = (2.0/m)*P.dot(P.T)
	
		return K



	def get_Wᵴ(self, X, σ, α):		
	# Wᵴ is a m x d matrix where m is the rff dimension, α repeats the number of Wᵴ matrices
	# X is a d x n matrix
		d = X.shape[0]
		Wᵴ = np.empty((0, d))

		for i in range(α):
			DX = self.mult_by_Rademacher_distribution(X)
			HDX = self.fwht(DX)
			DHDX = self.mult_by_Rademacher_distribution(HDX)
			HDHDX = self.fwht(DHDX)
			DHDHDX = self.mult_by_Rademacher_distribution(HDHDX)
			HDHDHDX = self.fwht(DHDHDX)
			Wᵴ_new = (np.sqrt(d)/σ)*HDHDHDX
			import pdb; pdb.set_trace()
			Wᵴ = np.vstack((Wᵴ, Wᵴ_new))
		return Wᵴ

	def mult_by_Rademacher_distribution(self, X):	#X should be d x n
		d = X.shape[0]
		R = 2*np.round(np.random.rand(d)) - 1
		R = np.reshape(R, (d,1))
		return R*X


	def fwht(self, x):
		"""In-place Fast Walsh–Hadamard Transform of array a."""
		a = x.T.tolist()
	
		if len(x.shape) == 1:
			h = 1
			while h < len(a):
				for i in range(0, len(a), h * 2):
					for j in range(i, i + h):
						x = a[j]
						y = a[j + h]
						a[j] = x + y
						a[j + h] = x - y
				h *= 2
		
			a = (1/np.sqrt(2))*np.array(a)
			a = np.reshape(a,(a.shape[0],1))
			return a
	
		elif len(x.shape) == 2:
			b = x.T.tolist()
	
			for a in b: 
				h = 1
				while h < len(a):
					for i in range(0, len(a), h * 2):
						for j in range(i, i + h):
							x = a[j]
							y = a[j + h]
							a[j] = x + y
							a[j + h] = x - y
					h *= 2
			
			b = (1/np.sqrt(2))*np.array(b).T
			return b

   
if __name__ == "__main__":
	np.set_printoptions(precision=2)
	np.set_printoptions(threshold=30)
	np.set_printoptions(linewidth=300)
	np.set_printoptions(suppress=True)
	np.set_printoptions(threshold=sys.maxsize)



	σ = 0.5
	m = 2000

	X = np.array([[0.2,1],[0,0],[0,1],[0,-1],[4,4],[4,5],[3,4],[4,3]], dtype='f')	
	start_time = perf_counter() 
	orf = ORF()
	rff_K = orf.SORF(X, σ, 100)
	#rff_K = orf.RFF(X, m, σ)
	time1 = (perf_counter() - start_time)


	γ = 1.0/(2*σ*σ)
	start_time = perf_counter() 
	rbk = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ)
	time2 = (perf_counter() - start_time)

	print(rff_K, '\n')
	print(rbk)

#	start_time = time.time() 
#	rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=2000)
#	Z = rbf_feature.fit_transform(X)
#	rff_K = Z.dot(Z.T)
#	rff_time = (time.time() - start_time)

