#!/usr/bin/env python

import numpy as np
import numpy.matlib
import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize			# version : 0.17
import matplotlib.pyplot as plt
from sklearn import preprocessing
from numpy import genfromtxt

np.set_printoptions(precision=4)
np.set_printoptions(threshold=3000)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)

class relative_kernel():
	def __init__(self, X):
		n = X.shape[0]
		if n < 50:
			num_of_samples = n
		else:
			num_of_samples = 50
			
		unique_X = np.unique(X, axis=0)
		#neigh = NearestNeighbors(num_of_samples)
		neigh = NearestNeighbors(num_of_samples, p=1)
	
		neigh.fit(unique_X)
		
		[dis, idx] = neigh.kneighbors(X, num_of_samples, return_distance=True)
		dis_inv = 1/dis[:,1:]
		idx = idx[:,1:]
		
		total_dis = np.sum(dis_inv, axis=1)
		total_dis = np.reshape(total_dis,(n, 1))
		total_dis = np.matlib.repmat(total_dis, 1, num_of_samples-1)
		dis_ratios = dis_inv/total_dis
	
		result_store_dictionary = {}
		σ_list = np.zeros((n,1))
		
		for i in range(n):
			if str(X[i,:]) in result_store_dictionary:
				σ = result_store_dictionary[str(X[i,:])] 
				σ_list[i] = σ
				continue
	
			dr = dis_ratios[i,:]
	
			Δ = unique_X[idx[i,:],:] - X[i,:]
			Δ2 = Δ*Δ
			d = np.sum(Δ2,axis=1)
			σ = np.sqrt(np.sum(dr*d))
			σ_list[i] = σ
	
			result_store_dictionary[str(X[i,:])] = σ
	
		self.σ_list = σ_list
		self.σ2 = σ_list.dot(σ_list.T)
		self.D = None
		self.X = X
		self.n = n

	def spectral(self, X, sigma, k):
		Vgamma = 1/(2*sigma*sigma)
		return SpectralClustering(k, gamma=Vgamma).fit_predict(X)

	def cluster_plot(self, X, allocation, all_radius=None):
		cmap = ['b', 'g', 'r', 'c', 'm', 'y','k']
	
		labels = np.unique(allocation)
		n = labels.shape[0]
	
		#plt.subplot(121)
		for i, j in enumerate(labels):
			subX = X[allocation == labels[i]]
			plt.plot(subX[:,0], subX[:,1], cmap[i] + '.')
			if all_radius is not None:
				for p in range(X.shape[0]):
					if np.random.rand() > 0.10:
						circle = plt.Circle((X[p,:][0],X[p,:][1]), radius=all_radius[p], fill=None) #, color=cmap[i]
						plt.gca().add_patch(circle)
	
	
		plt.xlabel('x')
		plt.ylabel('y')
		plt.title('Clustering Results')
		plt.show()


	def STSC(self, X, k):
		n = X.shape[0]
		D = sklearn.metrics.pairwise.pairwise_distances(X, metric='euclidean', n_jobs=1)
		sD = np.sort(D, axis=1)
		σ_list = np.reshape(sD[:,7], (n, 1))
		#σ_list = np.reshape(sD[:,20], (n, 1))
		σ_list[σ_list == 0] = 10
		
	
		σ2 = σ_list.dot(σ_list.T)
	
		K = np.exp(-(D*D)/(σ2))
		np.fill_diagonal(K,0)
	
		D_inv = 1.0/np.sqrt(np.sum(K, axis=1))
		Dv = np.outer(D_inv, D_inv)
		DKD = Dv*K
	
		[U, U_normalized] = L_to_U(DKD, k)
		allocation = KMeans(k).fit_predict(U_normalized)
	
		result = {}
		result['allocation'] = allocation
		result['σ_list'] = σ_list
		return result




	def L_to_U(self, L, k):
		eigenValues,eigenVectors = np.linalg.eigh(L)
	
		n2 = len(eigenValues)
		n1 = n2 - k
		U = eigenVectors[:, n1:n2]
		U_lambda = eigenValues[n1:n2]
		U_normalized = normalize(U, norm='l2', axis=1)
		
		return [U, U_normalized]

	def get_kernel(self, center_kernel=False):
		if self.D is None:
			self.D = sklearn.metrics.pairwise.pairwise_distances(self.X, metric='euclidean', n_jobs=1)
	
		K = np.exp(-(self.D*self.D)/(self.σ2))
		np.fill_diagonal(K,0)
	
		H = np.eye(self.n) - (1.0/self.n)*np.ones((self.n,self.n))
		D_inv = 1.0/np.sqrt(np.sum(K, axis=1))
		Dv = np.outer(D_inv, D_inv)
		self.Kx = DKD = Dv*K

		if center_kernel: self.Kx = HDKDH = H.dot(DKD).dot(H)
		return self.Kx


	def get_clustering_result(self, k, center_kernel=False):
		self.get_kernel(center_kernel=False)
		[U, U_normalized] = self.L_to_U(self.Kx, k)
		allocation = KMeans(k).fit_predict(U_normalized)
		return allocation






if __name__ == "__main__":

	#X = genfromtxt('../dataset/smiley.csv', delimiter=','); k = 3
	X = genfromtxt('../dataset/spiral_arm.csv', delimiter=','); k = 3
	#X = genfromtxt('../dataset/inner_rings.csv', delimiter=','); k = 3
	#X = genfromtxt('../dataset/noisy_two_clusters.csv', delimiter=','); k = 3
	#X = genfromtxt('../dataset/four_lines.csv', delimiter=','); k = 4
	
	X = preprocessing.scale(X)
	
	if True:
		rK = relative_kernel(X)
		labels = rK.get_clustering_result(k, center_kernel=True)
		#rK.cluster_plot(X, labels, rk.σ_list)
		rK.cluster_plot(X, labels)
	else:
		#labels = spectral(X, 1.0, k)
		result = STSC(X,k)
		#rK.cluster_plot(X, result['allocation'], result['σ_list'])
		rK.cluster_plot(X, result['allocation'])
	
	
	
	
	#----------------------------------------------------------------------
	
	
	#X = genfromtxt('../dataset/facial_85.csv', delimiter=','); k = 20
	#Y = genfromtxt('../dataset/facial_true_labels_624x960.csv', delimiter=','); 
	#X = genfromtxt('../dataset/breast-cancer.csv', delimiter=','); k = 2
	#Y = genfromtxt('../dataset/breast-cancer-labels.csv', delimiter=',')	# k = 2
	#X = genfromtxt('../dataset/wine.csv', delimiter=','); k = 3
	#Y = genfromtxt('../dataset/wine_label.csv', delimiter=',')	# k = 2
	
	#X = preprocessing.scale(X)
	#
	#if True:
	#	rK = relative_kernel(X)
	#	labels = rK.get_clustering_result(k, center_kernel=True)
	#	nmi = normalized_mutual_info_score(labels, Y)
	#else:
	#	result = STSC(X,k)
	#	##labels = spectral(X, 1.0, 20)
	#	nmi = normalized_mutual_info_score(result['allocation'], Y)
	#
	#print(nmi)
	
	
	
	#----------------------------------------------------------------------
	
	##n = 1000
	#x1 = np.random.randn(6,2) + np.array([4,4]); k = 2
	#x2 = np.random.randn(6,2) + np.array([-4,-4])
	#X = np.vstack((x1,x2))
	#
	##result = STSC_σ(X,k)
	#result = STSC(X,k)
	#
	#rK.cluster_plot(X, result['allocation'], result['σ_list'])
	
