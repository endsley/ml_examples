#!/usr/bin/env python

import sys
import numpy as np
import sklearn.metrics
import torch
import numpy.matlib
from torch.autograd import Variable
from numpy.random import rand
import torch.nn.functional as F

np.set_printoptions(precision=3)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)


class sparse_pca:
	def __init__(self):
		self.Kᵭ_threshold = 0.035 		# kernel divergence threshold
		self.debug_record = {}

	def subsample_matrix_with_dim(self, A, num_of_eigs, eigs_1, λ_id, ń): #A is square, n is the number we keep
		N = A.shape[0]
		inc = 1
		Kᵭ = 10
		eigs_2 = 0

		for m in range(400):
			new_arrangement = np.random.permutation(N)
			keep_index_better = np.sort(new_arrangement[0:ń])
			discard_index_better = np.sort(new_arrangement[ń:])
	
			B = A[keep_index_better, :]
			Aᵴ_better = B[:, keep_index_better]
		
			[D,V] = np.linalg.eigh(Aᵴ_better)
			eigs_2_better = D[-1]
			v2_better = V[:,-1]
			
			Kᵭ_better = np.linalg.norm(eigs_1 - eigs_2_better, ord=np.inf)
			if eigs_2_better > eigs_2:
				Kᵭ = Kᵭ_better
				Aᵴ = Aᵴ_better
				v2 = v2_better
				keep_index = keep_index_better
				eigs_2 = eigs_2_better
				discard_index = discard_index_better
				#print(eigs_2)

		self.debug_record[λ_id][Kᵭ] = Kᵭ
		return [Aᵴ, v2, keep_index, discard_index]




	def subsample_matrix(self, A, num_of_eigs, eigs_1, λ_id): #A is square, n is the number we keep
		N = A.shape[0]
		inc = 1

		for ń in np.arange(num_of_eigs, A.shape[0]+1, inc):
			new_arrangement = np.random.permutation(N)
			keep_index = np.sort(new_arrangement[0:ń])
			discard_index = np.sort(new_arrangement[ń:])
			
			B = A[keep_index, :]
			Aᵴ = B[:, keep_index]
		
			[ksf_2, eigs_2, num_of_eigs, v2, keepK_num] = self.get_kernel_sampling_feature(Aᵴ, num_of_eigs)
			
			if len(eigs_1) > len(eigs_2):	
				extra_pad = len(eigs_1) - len(eigs_2)
				eigs_2 = np.pad(eigs_2, (0,extra_pad), 'constant')
				
			Kᵭ = np.linalg.norm(eigs_1 - eigs_2, ord=np.inf)
			if Kᵭ < self.Kᵭ_threshold: 
				for m in range(20):

					new_arrangement = np.random.permutation(N)
					keep_index_better = np.sort(new_arrangement[0:ń])
					discard_index_better = np.sort(new_arrangement[ń:])
		
					B = A[keep_index_better, :]
					Aᵴ_better = B[:, keep_index_better]
				
					[ksf_2_better, eigs_2_better, num_of_eigs_better, v2_better, keepK_num] = self.get_kernel_sampling_feature(Aᵴ_better, num_of_eigs)
					
					if len(eigs_1) > len(eigs_2_better):	
						extra_pad = len(eigs_1) - len(eigs_2_better)
						eigs_2_better = np.pad(eigs_2_better, (0,extra_pad), 'constant')
						
					Kᵭ_better = np.linalg.norm(eigs_1 - eigs_2_better, ord=np.inf)
					if Kᵭ_better < Kᵭ:
						Kᵭ = Kᵭ_better
						Aᵴ = Aᵴ_better
						v2 = v2_better
						keep_index = keep_index_better
						discard_index = discard_index_better

				self.debug_record[λ_id][Kᵭ] = Kᵭ
				return [Aᵴ, v2, keep_index, discard_index]




	def double_center(self, Ψ):
		HΨ = Ψ - np.mean(Ψ, axis=0)								# equivalent to Γ = Ⲏ.dot(Kᵧ).dot(Ⲏ)
		HΨH = (HΨ.T - np.mean(HΨ.T, axis=0)).T
		return HΨH


	def get_kernel_sampling_feature(self, rbk, num_of_eigs=None):
		[U,S,V] = np.linalg.svd(rbk)

		v = V[0,:]	# most dominant eig
		sorted_v = np.flip(np.sort(np.absolute(v)))
		keepK_num = np.sum(np.cumsum(sorted_v/np.sum(sorted_v)) < self.sparcity)

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

	def get_sparce_eig_vector(self, A, num_of_eigs, sparcity=0.9):
		orig_A = A.copy()
		self.sparcity = sparcity		# 0 to 1, the closer to 0 the more sparse
		n = A.shape[0]
		eigvec = np.empty((n, 0))
		
		for λ_id in np.arange(1, num_of_eigs + 1):
			print('Layer : %d'%λ_id)
			self.debug_record[λ_id] = {}
	
			[keepCS, eigs_1, num_of_eigs, v1, keepK_num] = self.get_kernel_sampling_feature(A, 1)
			[Aᵴ, v2, keep_index, discard_index] = self.subsample_matrix_with_dim(A, num_of_eigs, eigs_1, λ_id, keepK_num)

			vi = self.merge_eig_vectors(v1,v2, A, Aᵴ, keep_index, discard_index, λ_id)
			vi = np.reshape(vi, (len(vi), 1))
	
			for m in np.arange(eigvec.shape[1]):
				U = np.reshape(eigvec[:,m], (n,1))
				vi = vi - U.T.dot(vi)*U

			vi = vi/np.linalg.norm(vi)
			self.debug_1(λ_id, v1, v2, vi, Aᵴ, A)
	
			eigvec = np.hstack((eigvec, vi))
			A = self.matrix_projection_deflation(A, vi)

		D = (np.diag(eigvec.T.dot(orig_A).dot(eigvec)))
		return [D, eigvec]


#x = np.vstack((np.random.randn(30,2), np.random.randn(30,2) + 5))
#A = x.dot(x.T)
#A = np.random.randn(100,100)
#A = A.dot(A.T)
A =  np.array([[ 5.95 ,  2.558, -0.045,  5.369,  0.571,  4.871, -1.672,  0.784, -1.175, -2.351],
				[ 2.558,  9.927, -1.97 ,  1.867,  8.123,  1.998, -3.867, -5.423, -4.889, -3.455],
				[-0.045, -1.97 , 11.314,  3.922, -3.707,  2.685,  5.36 , -4.995, -0.136,  1.396],
				[ 5.369,  1.867,  3.922,  8.214,  0.626,  3.814, -0.26 , -2.162,  1.232, -1.246],
				[ 0.571,  8.123, -3.707,  0.626, 12.372, -0.182, -2.772, -1.065, -2.313,  0.384],
				[ 4.871,  1.998,  2.685,  3.814, -0.182,  6.045,  0.541,  0.586, -3.12 , -1.502],
				[-1.672, -3.867,  5.36 , -0.26 , -2.772,  0.541,  4.191,  0.548,  0.72 ,  2.633],
				[ 0.784, -5.423, -4.995, -2.162, -1.065,  0.586,  0.548, 10.719,  2.188,  2.767],
				[-1.175, -4.889, -0.136,  1.232, -2.313, -3.12 ,  0.72 ,  2.188,  5.608,  2.17 ],
				[-2.351, -3.455,  1.396, -1.246,  0.384, -1.502,  2.633,  2.767,  2.17 ,  3.465]])

#[D,V] = np.linalg.eigh(A)
#print(np.flip(np.sort(D)))
sp = sparse_pca()
[D,V] = sp.get_sparce_eig_vector(A,5, sparcity=0.7)		# 0 to 1, the closer to 0 the more sparse
print('Eigen values : ', D)
print('Eigen vector : ')
print(V)
