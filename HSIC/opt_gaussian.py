#!/usr/bin/env python


import sys
import matplotlib
import numpy as np
import random
import itertools
import socket
import sklearn.metrics
from scipy.optimize import minimize
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.random import sample_without_replacement

np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)



class opt_gaussian():
	def __init__(self, X, Y, Y_kernel='linear'):	#	X=data, Y=label, σ_type='ℍ', or 'maxKseparation'
		ń = X.shape[0]
		ð = X.shape[1]

		if ń > 200:
			#	Down sample first
			samples = sample_without_replacement(n_population=ń, n_samples=200)
			X = X[samples,:]
			Y = Y[samples]
			ń = X.shape[0]


		if Y_kernel == 'linear':
			self.Ⅱᵀ = np.ones((ń,ń))
			ńᒾ = ń*ń
			Yₒ = OneHotEncoder(categories='auto', sparse=False).fit_transform(np.reshape(Y,(len(Y),1)))
			self.Kᵧ = Kᵧ = Yₒ.dot(Yₒ.T)
			ṉ = np.sum(Kᵧ)
		elif Y_kernel == 'Gaussian'
			Ðᵧ = sklearn.metrics.pairwise.pairwise_distances(Y)
			σᵧ = np.median(Ðᵧ)
			self.Ðᵧᒾ = (-Ðᵧ*Ðᵧ)/2


		#	X_kernel == 'Gaussian'
		Ðₓ = sklearn.metrics.pairwise.pairwise_distances(X)
		σₓ = np.median(Ðₓ)
		self.Ðₓᒾ = (-Ðₓ*Ðₓ)/2

		σ = [σₓ, σᵧ]


		self.result = minimize(self.ℍ, σ, method='BFGS', options={'gtol': 1e-5, 'disp': False})

	def ℍ(self, σ):
		[σₓ, σᵧ] = σ
		Kₓ = np.exp(self.Ðₓᒾ/(σₓ*σₓ))
		Kᵧ = np.exp(self.Ðᵧᒾ/(σᵧ*σᵧ))
		HKᵧ = Kᵧ - np.mean(Kᵧ, axis=0)								# equivalent to Γ = Ⲏ.dot(Kᵧ).dot(Ⲏ)
		Γ = HKᵧH = (HKᵧ.T - np.mean(HKᵧ.T, axis=0)).T

		loss = -np.sum(Kₓ*Γ)
		return loss



	def debug(self):
		ℍ_debug(self)


def get_opt_σ(X,Y, Y_kernel='Gaussian'):
	optimizer = opt_gaussian(X,Y, Y_kernel=Y_kernel)
	return optimizer.result.x



if __name__ == "__main__":
	data_name = 'wine_2'
	X = np.loadtxt('../dataset/' + data_name + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt('../dataset/' + data_name + '_label.csv', delimiter=',', dtype=np.int32)			
	X = preprocessing.scale(X)


	opt_σ = opt_gaussian(X,Y)	#σ_type='ℍ' or 'maxKseparation'
	import pdb; pdb.set_trace()

	#opt_σ.debug()

