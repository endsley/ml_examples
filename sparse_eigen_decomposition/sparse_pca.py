#!/usr/bin/env python

import sys
import numpy as np
import sklearn.metrics
import torch
import numpy.matlib
from numpy.random import rand
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans


np.set_printoptions(precision=3)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

# documentation : 
#https://github.com/endsley/math_notebook/blob/master/MyPapers/Multi-source_clustering%2BSparse_PCA/Multi_source_Clustering_sparse_PCA.pdf

class sparse_pca:
	def __init__(self):
		self.Kᵭ_threshold = 0.035 		# kernel divergence threshold
		self.debug_record = {}

	def subsample_matrix_with_dim(self, A, eigs_1, λ_id, ń): #A is square, n is the number we keep
		N = A.shape[0]
		inc = 1
		Kᵭ = 10
		eigs_2 = 0

		for ń in np.arange(ń, N, inc):
			new_arrangement = np.random.permutation(N)
			keep_index_better = np.sort(new_arrangement[0:ń])
			discard_index_better = np.sort(new_arrangement[ń:])
	
			B = A[keep_index_better, :]
			Aᵴ_better = B[:, keep_index_better]
		
			[D,V] = np.linalg.eigh(Aᵴ_better)
		
			try:
				eigs_2_better = D[-1]
			except:
				import pdb; pdb.set_trace()

			v2_better = V[:,-1]
			
			if eigs_2_better > eigs_2:
				Aᵴ = Aᵴ_better
				v2 = v2_better
				keep_index = keep_index_better
				eigs_2 = eigs_2_better
				discard_index = discard_index_better

				#print(eigs_1, eigs_2)
				if eigs_1*0.999 < eigs_2:
					print('\tEigen percentage : %.3f,  %.3f/%.3f'% (eigs_2/eigs_1, eigs_2, eigs_1))
					print('\tSamples used : %d'% (ń))
					break
					#import pdb; pdb.set_trace()

		self.debug_record[λ_id][Kᵭ] = Kᵭ
		return [Aᵴ, v2, keep_index, discard_index]


	def get_kernel_sampling_feature(self, rbk, num_of_eigs=None):
		[U,S,V] = np.linalg.svd(rbk)

		v = V[0,:]	# most dominant eig
		sorted_v = np.flip(np.sort(np.absolute(v)))
		keepK_num = np.sum(np.cumsum(sorted_v/np.sum(sorted_v)) < self.sparcity)
		print('\t', keepK_num)

		cs = np.cumsum(S)/np.sum(S)
		L1_norm = S/np.sum(S)
		
		if num_of_eigs is None: num_of_eigs = np.sum(cs < 0.9) + 1
		keepCS = cs[0:num_of_eigs]
		L1_norm = L1_norm[0:num_of_eigs]

		return [keepCS, L1_norm, num_of_eigs, v, keepK_num]

	def merge_eig_vectors(self, v1, v2, A, Aᵴ, keep_index, discard_index, λ_id):
		n = len(v2)
		v_new = v1.copy()
		v_new[discard_index] = 0
		v_new[keep_index] = v2

		return v_new
	
	def matrix_projection_deflation(self, A, v):
		n = v.shape[0]
		I = np.eye(n)
		A = (I - v.dot(v.T)).dot(A).dot(I - v.dot(v.T))
		return A

	def debug_1(self, λ_id, v1, v2, v_new, Aᵴ, A):
		self.debug_record[λ_id]['vAv'] = v1.T.dot(A).dot(v1)
		self.debug_record[λ_id]['vAᵴv'] = v2.T.dot(Aᵴ).dot(v2)
		self.debug_record[λ_id]['ῡAῡ'] = v_new.T.dot(A).dot(v_new)

		print('\t vAv : %.3f'%self.debug_record[λ_id]['vAv'])
		print('\t vAᵴv : %.3f'%self.debug_record[λ_id]['vAᵴv'])
		print('\t ῡAῡ : %.3f'%self.debug_record[λ_id]['ῡAῡ'])

	def get_sparce_eig_vector(self, A, num_of_eigs=None, sparcity=0.95):
		orig_A = A.copy()
		self.sparcity = sparcity		# 0 to 1, the closer to 0 the more sparse
		n = A.shape[0]
		eigvec = np.empty((n, 0))
		
		if num_of_eigs is None: num_of_eigs = 1000
		for λ_id in np.arange(1, num_of_eigs + 1):
			print('Layer : %d'%λ_id)
			self.debug_record[λ_id] = {}
	
			#[keepCS, eigs_1, num_of_eigs, v1, keepK_num] = self.get_kernel_sampling_feature(A, 1)


			[D,V] = np.linalg.eigh(A)
			eigs_1 = D[-1]
			v1 = V[:,-1]
			sorted_v = np.flip(np.sort(np.absolute(v1)))
			keepK_num = np.sum(np.cumsum(sorted_v/np.sum(sorted_v)) < self.sparcity)
			

			[Aᵴ, v2, keep_index, discard_index] = self.subsample_matrix_with_dim(A, eigs_1, λ_id, keepK_num)

			vi = self.merge_eig_vectors(v1,v2, A, Aᵴ, keep_index, discard_index, λ_id)
			vi = np.reshape(vi, (len(vi), 1))
	
			for m in np.arange(eigvec.shape[1]):
				U = np.reshape(eigvec[:,m], (n,1))
				vi = vi - U.T.dot(vi)*U

			vi = vi/np.linalg.norm(vi)
			self.debug_1(λ_id, v1, v2, vi, Aᵴ, A)
	
			eigvec = np.hstack((eigvec, vi))
			A = self.matrix_projection_deflation(A, vi)

			if num_of_eigs == 1000:
				num_not_assigned = np.sum(np.sum(np.absolute(eigvec), axis=1) < 0.0000001)
				print('\tNumber of samples not assigned : %d'%num_not_assigned)
				if num_not_assigned == 0:
					D = (np.diag(eigvec.T.dot(orig_A).dot(eigvec)))
					return [D, eigvec]

		D = (np.diag(eigvec.T.dot(orig_A).dot(eigvec)))
		return [D, eigvec]


if __name__ == "__main__":
	#	Data 1
	data_name = 'wine'
	#data_name = 'cancer'
	#data_name = 'car'

	X = np.loadtxt('data/' + data_name + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt('data/' + data_name + '_label.csv', delimiter=',')			
	Y = np.reshape(Y, (len(Y),1))
	X = preprocessing.scale(X)
	#γ = 1.0/(2*0.3*0.3)
	γ = 1.0/(2)
	A = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ)
	np.fill_diagonal(A, 0)			#	Set diagonal of adjacency matrix to 0

	##	Data 2
	#x = np.vstack((np.random.randn(10,5), np.random.randn(10,5) + 5, np.random.randn(10,5) + 10))
	#x = preprocessing.scale(x)
	#γ = 1.0/(2*0.5*0.5)
	#A = sklearn.metrics.pairwise.rbf_kernel(x, gamma=γ)
	#np.fill_diagonal(A, 0)			#	Set diagonal of adjacency matrix to 0

	#	Input matrix A must be SPSD
	sp = sparse_pca()
	[D,V] = sp.get_sparce_eig_vector(A, num_of_eigs=None)		# num_of_eigs : how many eigen vectors to produce, None automatically choose minimum
	[D2,V2] = np.linalg.eigh(A)
	
	
	print('\nOutput')
	print('Actual Eigen values : ', np.flip(D2)[0:V.shape[1]])
	print('Sparse Eigen values : ', D)

	print('\nNotice how using sparse eigenvectors achieves the similar magnitude')
	print('\t vAv : %.3f'% np.sum(np.flip(D2)[0:V.shape[1]]))
	print('\t ῡAῡ : %.3f'%np.trace(V.T.dot(A).dot(V)))
	import pdb; pdb.set_trace()

	#print('Eigen vector : ')
	#print(V)

