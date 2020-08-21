#!/usr/bin/env python

import numpy as np
import sklearn.metrics
import sklearn.metrics.pairwise
from scipy.stats.stats import pearsonr  
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

def ℍ(X,Y, X_kernel='Gaussian', Y_kernel='Gaussian'):	# compute normalized HSIC between X,Y
	if len(X.shape) == 1: X = np.reshape(X, (X.size, 1))
	if len(Y.shape) == 1: Y = np.reshape(Y, (Y.size, 1))
	n = X.shape[0]

	if X_kernel == 'linear': Kᵪ = X.dot(X.T)
	if Y_kernel == 'linear': Kᵧ = Y.dot(Y.T)

	if X_kernel == 'Gaussian': 
		σ = np.median(sklearn.metrics.pairwise_distances(X))
		γ = 1.0/(2*σ*σ)
		Kᵪ = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ)

	if Y_kernel == 'Gaussian': 
		σ = np.median(sklearn.metrics.pairwise_distances(Y))
		γ = 1.0/(2*σ*σ)
		Kᵧ = sklearn.metrics.pairwise.rbf_kernel(Y, gamma=γ)


	HKᵪ = Kᵪ - np.mean(Kᵪ, axis=0)					# equivalent to		HKᵪ = H.dot(Kᵪ)
	HKᵧ = Kᵧ - np.mean(Kᵧ, axis=0)                  # equivalent to		HKᵧ = H.dot(Kᵧ)

	Hᵪᵧ= np.sum(HKᵪ*HKᵧ)

	Hᵪ = np.linalg.norm(HKᵪ)						# equivalent to 	np.sqrt(np.sum(KᵪH*KᵪH))
	Hᵧ = np.linalg.norm(HKᵧ) 						# equivalent to 	np.sqrt(np.sum(KᵧH*KᵧH))
	H = Hᵪᵧ/( Hᵪ * Hᵧ )

	return H



def double_center(Ψ):
	HΨ = Ψ - np.mean(Ψ, axis=0)								# equivalent to Γ = Ⲏ.dot(Kᵧ).dot(Ⲏ)
	HΨH = (HΨ.T - np.mean(HΨ.T, axis=0)).T
	return HΨH


if __name__ == '__main__':
	n = 300
	
	#	Linear Data
	dat = np.random.rand(n,1)
	linear_data = np.hstack((dat,dat)) + 0.04*np.random.randn(n,2)
	linear_pc = np.round(pearsonr(linear_data[:,0], linear_data[:,1])[0], 2)
	linear_nmi = normalized_mutual_info_score(linear_data[:,0], linear_data[:,1])
	linear_hsic = np.round(ℍ(linear_data[:,0], linear_data[:,1]),2)
	
	print('Linear Relationship:')
	print('\tCorrelation : ', linear_pc)
	print('\tNMI : ', linear_nmi)
	print('\tHSIC : ', linear_hsic)

	#	Sine Data
	dat_x = 9.3*np.random.rand(n,1)
	dat_y = np.sin(dat_x)
	sine_data = np.hstack((dat_x,dat_y)) + 0.06*np.random.randn(n,2)
	sine_pc = np.round(pearsonr(sine_data[:,0], sine_data[:,1])[0],2)
	sine_nmi = normalized_mutual_info_score(sine_data[:,0], sine_data[:,1])
	sine_hsic = np.round(ℍ(sine_data[:,0], sine_data[:,1]),2)
	
	print('Sine Relationship:')
	print('\tCorrelation : ', sine_pc)
	print('\tNMI : ', sine_nmi)
	print('\tHSIC : ', sine_hsic)


	#	Parabola Data
	dat_x = 4*np.random.rand(n,1) - 2
	dat_y = 0.05*dat_x*dat_x
	para_data = np.hstack((dat_x,dat_y)) + 0.01*np.random.randn(n,2)
	#para_data[n/2:, 1] = -para_data[n/2:, 1] + 0.4	

	para_pc = np.round(pearsonr(para_data[:,0], para_data[:,1])[0],2)
	para_nmi = normalized_mutual_info_score(para_data[:,0], para_data[:,1])
	para_hsic = np.round(ℍ(para_data[:,0], para_data[:,1]),2)
	
	print('Parabola Relationship:')
	print('\tCorrelation : ', para_pc)
	print('\tNMI : ', para_nmi)
	print('\tHSIC : ', para_hsic)


	#	Random uniform Data
	unif_data = np.random.rand(n,2)
	unif_pc = np.round(pearsonr(unif_data[:,0], unif_data[:,1])[0],2)
	unif_hsic = np.round(ℍ(unif_data[:,0], unif_data[:,1]),2)
	unif_nmi = normalized_mutual_info_score(unif_data[:,0], unif_data[:,1])
	
	print('Random Relationship:')
	print('\tCorrelation : ', unif_pc)
	print('\tNMI : ', unif_nmi)
	print('\tHSIC : ', unif_hsic)



	plt.figure(figsize=(9,2))


	plt.subplot(141)
	plt.plot(linear_data[:,0], linear_data[:,1], 'bx')
	plt.title('$\\rho$ : ' + str(linear_pc) + ' , HSIC : ' + str(linear_hsic))
	
	plt.subplot(142)
	plt.plot(sine_data[:,0], sine_data[:,1], 'bx')
	plt.title('$\\rho$ : ' + str(sine_pc) + ' , HSIC : ' + str(sine_hsic))

	plt.subplot(143)
	plt.plot(para_data[:,0], para_data[:,1], 'bx')
	plt.title('$\\rho$ : ' + str(para_pc) + ' , HSIC : ' + str(para_hsic))

	plt.subplot(144)
	plt.plot(unif_data[:,0], unif_data[:,1], 'bx')
	plt.title('$\\rho$ : ' + str(unif_pc) + ' , HSIC : ' + str(unif_hsic))

	plt.tight_layout()
	plt.show()


