#!/usr/bin/env python


import sys
import matplotlib
import numpy as np
import random
import itertools
import socket
import sklearn.metrics
from scipy.optimize import minimize
from scipy.optimize import Bounds
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.random import sample_without_replacement

np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)



class opt_gaussian():
	def __init__(self, X, Y, Y_kernel):	#	X=data, Y=label, σ_type='ℍ', or 'maxKseparation'
		ń = X.shape[0]
		ð = X.shape[1]
		self.Y_kernel = Y_kernel

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
			σᵧ = 1

			HKᵧ = self.Kᵧ - np.mean(self.Kᵧ, axis=0)								# equivalent to Γ = Ⲏ.dot(Kᵧ).dot(Ⲏ)
			self.Γ = HKᵧH = (HKᵧ.T - np.mean(HKᵧ.T, axis=0)).T

		elif Y_kernel == 'Gaussian':
			Ðᵧ = sklearn.metrics.pairwise.pairwise_distances(Y)
			σᵧ = np.median(Ðᵧ)
			self.Ðᵧᒾ = (-Ðᵧ*Ðᵧ)/2


		#	X_kernel == 'Gaussian'
		Ðₓ = sklearn.metrics.pairwise.pairwise_distances(X)
		σₓ = np.median(Ðₓ)
		self.Ðₓᒾ = (-Ðₓ*Ðₓ)/2

		self.σ = [σₓ, σᵧ]

	def minimize_H(self):
		self.result = minimize(self.ℍ, self.σ, method='L-BFGS-B', options={'gtol': 1e-5, 'disp': False}, bounds=Bounds(0.05, 100000))


	def ℍ(self, σ):
		[σₓ, σᵧ] = σ
		Kₓ = np.exp(self.Ðₓᒾ/(σₓ*σₓ))


		if self.Y_kernel == 'linear':
			Γ = self.Γ		
		elif self.Y_kernel == 'Gaussian':
			Kᵧ = np.exp(self.Ðᵧᒾ/(σᵧ*σᵧ))
			HKᵧ = Kᵧ - np.mean(Kᵧ, axis=0)								# equivalent to Γ = Ⲏ.dot(Kᵧ).dot(Ⲏ)
			Γ = HKᵧH = (HKᵧ.T - np.mean(HKᵧ.T, axis=0)).T


		loss = -np.sum(Kₓ*Γ)
		return loss


def get_opt_σ(X,Y, Y_kernel='Gaussian'):
	optimizer = opt_gaussian(X,Y, Y_kernel=Y_kernel)
	optimizer.minimize_H()
	return optimizer.result

def get_opt_σ_via_random(X,Y, Y_kernel='Gaussian'):
	optimizer = opt_gaussian(X,Y, Y_kernel=Y_kernel)

	opt = 0
	opt_σ = 0
	for m in range(1000):
		σ = (7*np.random.rand(2)).tolist()
		new_opt = -optimizer.ℍ(σ)
		if opt < new_opt:
			opt = new_opt
			opt_σ = σ

	print('Random Result ')
	print('\tbest_σ : ', opt_σ)
	print('\tmax_HSIC : ' , opt)


if __name__ == "__main__":
	data_name = 'wine'
	X = np.loadtxt('../dataset/' + data_name + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt('../dataset/' + data_name + '_label.csv', delimiter=',', dtype=np.int32)			
	X = preprocessing.scale(X)


	optimized_results = get_opt_σ(X,Y, Y_kernel='linear')
	best_σ = optimized_results.x
	max_HSIC = -optimized_results.fun
	print('Optimized Result ')
	print('\tbest_σ [σₓ, σᵧ]: ', best_σ)
	print('\tmax_HSIC : ' , max_HSIC)


	optimized_results = get_opt_σ_via_random(X,Y, Y_kernel='linear')


