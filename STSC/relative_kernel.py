

import numpy as np
import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize			# version : 0.17

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

