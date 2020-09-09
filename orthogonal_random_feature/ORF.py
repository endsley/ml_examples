#!/usr/bin/env python

import numpy as np
import sys
import numpy.matlib
import sklearn.metrics
from time import perf_counter
from scipy.linalg import hadamard
from numpy import genfromtxt
from sklearn import preprocessing
import sklearn.metrics
import pyrfm.random_feature

class ORF():
	def __init__(self):
		pass

	def pad_X_to_power_of_2(self, X):
		n = X.shape[0]
		d = X.shape[1]
		next_power = np.ceil(np.log2(d))

		if next_power == d: return X

		nP = np.power(2, next_power)
		Δ = int(nP - d)
		Z = np.zeros((n,Δ))
		X =  np.hstack((X,Z))
		return X

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
		γ = 1.0/(2*σ*σ)
		sorf = pyrfm.random_feature.StructuredOrthogonalRandomFeature(gamma=γ)
		Φx = sorf.fit_transform(X)
		K = Φx.dot(Φx.T)

#		My version is not working, I think it is because I am using shift instead of cosine sine

#		X = self.pad_X_to_power_of_2(X)
#
#		N = X.shape[0]
#		d = X.shape[1]
#		m = d*α	
#
#		phase_shift = 2*np.pi*np.random.rand(1, m)
#		phase_shift = np.matlib.repmat(phase_shift, N, 1)
#		#WX = self.get_WX(X.T, σ, α)
#		WX = self.get_WX_slow(X.T, σ, α)
#		#import pdb; pdb.set_trace()	
#
#		P = np.cos(WX.T + phase_shift)
#		K = (2.0/m)*P.dot(P.T)
#	
#		return K


	def get_WX_slow(self, X, σ, α):		
		d = X.shape[0]
		n = X.shape[1]


		#Qx = np.empty((0, d))
		#for i in range(α):
		#	D = np.diag(2*np.round(np.random.rand(d)) - 1)
	
		#	HD = self.mult_by_Hadamard_slow(D)
		#	DHD = self.mult_by_Rademacher_distribution(HD)
		#	HDHD = self.mult_by_Hadamard_slow(DHD)
		#	DHDHD = self.mult_by_Rademacher_distribution(HDHD)
		#	HDHDHD = self.mult_by_Hadamard_slow(DHDHD)
	
		#	Q_new = (np.sqrt(d)/σ)*HDHDHD
		#	Qx = np.vstack((Qx, Q_new))

		#import pdb; pdb.set_trace()


		Qx = np.empty((0, n))
		for i in range(α):
			DX = self.mult_by_Rademacher_distribution(X)
			HDX = self.mult_by_Hadamard_slow(DX)
			DHDX = self.mult_by_Rademacher_distribution(HDX)
			HDHDX = self.mult_by_Hadamard_slow(DHDX)
			DHDHDX = self.mult_by_Rademacher_distribution(HDHDX)
			HDHDHDX = self.mult_by_Hadamard_slow(DHDHDX)
			Q_new = (np.sqrt(d)/σ)*HDHDHDX
			Qx = np.vstack((Qx, Q_new))
		return Qx

	def mult_by_Hadamard_slow(self, x):	# x is in column form
		l = x.shape[0]
		H = hadamard(l)
		#return H.dot(x)
		return (1/np.sqrt(2))*H.dot(x)


	def get_WX(self, X, σ, α):
	# Qx is a d x n matrix where m is the rff dimension, α repeats the number of Wᵴ matrices
	# X is a d x n matrix
		d = X.shape[0]
		n = X.shape[1]
		Qx = np.empty((0, n))

		for i in range(α):
			DX = self.mult_by_Rademacher_distribution(X)
			HDX = self.fwht(DX)
			DHDX = self.mult_by_Rademacher_distribution(HDX)
			HDHDX = self.fwht(DHDX)
			DHDHDX = self.mult_by_Rademacher_distribution(HDHDX)
			HDHDHDX = self.fwht(DHDHDX)
			Q_new = (np.sqrt(d)/σ)*HDHDHDX
			Qx = np.vstack((Qx, Q_new))

		return Qx

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


	#X = np.array([[0.2,1],[0,0],[0,1],[0,-1],[4,4],[4,5],[3,4],[4,3]], dtype='f')	
	#X1 = np.random.randn(30,16)
	#X2 = np.random.randn(30,16) + 3
	#X = np.vstack((X1,X2))

	X = genfromtxt('../dataset/wine.csv', delimiter=',')
	X = preprocessing.scale(X)

	σ = np.median(sklearn.metrics.pairwise.pairwise_distances(X))
	α = 100
	m = X.shape[1]*α


	start_time = perf_counter() 
	orf = ORF()
	orf_K = orf.SORF(X, σ, 100)
	time1 = (perf_counter() - start_time)

	start_time = perf_counter() 
	rff_K = orf.RFF(X, m, σ)
	time1 = (perf_counter() - start_time)



	γ = 1.0/(2*σ*σ)
	start_time = perf_counter() 
	rbk = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ)
	time2 = (perf_counter() - start_time)

	print('ORF : \n', orf_K[0:10,0:10], '\n')
	print('RFF : \n', rff_K[0:10,0:10], '\n')
	print('RBK : \n', rbk[0:10,0:10])

#	start_time = time.time() 
#	rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=2000)
#	Z = rbf_feature.fit_transform(X)
#	rff_K = Z.dot(Z.T)
#	rff_time = (time.time() - start_time)

