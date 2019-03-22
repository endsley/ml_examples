
import numpy as np
import sklearn.metrics


class subset_select():
	def __init__(self, X):
		if isinstance(X, str):
			self.X = np.loadtxt(X, delimiter=',', dtype=np.float64)
		else:
			self.X = X
		self.threashold = 0.01

	def rbk_sklearn(self, data, σ, center=False):
		γ = 1.0/(2*σ*σ)
		rbk = sklearn.metrics.pairwise.rbf_kernel(data, gamma=γ)
		N = rbk.shape[0]
		H = np.eye(N) - np.ones((N,N))/float(N)
		rbk = H.dot(rbk).dot(H)


		return rbk

	def save_subset_to_file(self, name, Y=None):
		filename = name + '.csv'
		np.savetxt(filename, self.new_X, delimiter=',', fmt='%.3f') 

		if Y is not None:
			if isinstance(Y, str):
				Y = np.loadtxt(Y, delimiter=',', dtype=np.float64)

			new_Y = Y[self.best_test_sample_id]
			filename = name + '_label.csv'
			np.savetxt(filename, new_Y, delimiter=',', fmt='%.1f') 



	def get_subset(self):
		σ = np.median(sklearn.metrics.pairwise.pairwise_distances(self.X))
		N = self.X.shape[0]
		K_orig = self.rbk_sklearn(self.X, σ)
		[D,V] = np.linalg.eigh(K_orig)

		scaled_cumsum_D = np.cumsum(np.flip(D,0)/np.sum(D))
		eigLen = len(scaled_cumsum_D[scaled_cumsum_D < 0.95])
		largest_eigs = np.flip(D,0)[0:eigLen]
		largest_eigs = largest_eigs/np.sum(largest_eigs)

		for test_percent in np.arange(0.05,0.9,0.05):
			kd_list = []
			lowest_Kd = 100
			best_test_sample_id = None
			for rep in range(10):
				inc = int(np.floor(test_percent*N))
				if inc < eigLen: continue
		
				rp = np.random.permutation(N).tolist()
				test_set_id = rp[0:inc]
				sample_X = self.X[test_set_id,:]
		
				K_new = self.rbk_sklearn(sample_X, σ)

				[D,V] = np.linalg.eigh(K_new)
				small_eigs = np.flip(D,0)[0:eigLen]
				small_eigs = small_eigs/np.sum(small_eigs)
		
				Kd = np.max(np.absolute(largest_eigs - small_eigs))
				kd_list.append(Kd)
	
				if Kd < lowest_Kd:
					lowest_Kd = Kd
					#print(lowest_Kd)
					best_test_sample_id = test_set_id
					test_set_indx = list(set(rp) - set(best_test_sample_id))
		
			avg_kd = np.mean(kd_list)
			print('At %.3f percent, avg error : %.3f'%(test_percent, avg_kd))
			if avg_kd < self.threashold: break
	
		self.best_test_sample_id = best_test_sample_id
		self.new_X = self.X[best_test_sample_id,:]
		K_new = self.rbk_sklearn(self.new_X, σ)

		[D,V] = np.linalg.eigh(K_new)
		small_eigs = np.flip(D,0)[0:eigLen]
		small_eigs = small_eigs/np.sum(small_eigs)
		Kd = np.max(np.absolute(largest_eigs - small_eigs))
		print('\n%.3f percent was chosen with kernel divergence error of %.3f'%(test_percent, Kd))

		return [self.new_X, best_test_sample_id]

