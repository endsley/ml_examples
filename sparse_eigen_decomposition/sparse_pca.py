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
		pass


	def subsample_matrix(self, A, num_of_eigs, eigs_1): #A is square, n is the number we keep
		N = A.shape[0]
		inc = 1

		for ń in np.arange(num_of_eigs, A.shape[0]+1, inc):
			keep_index = np.random.permutation(N)[0:ń]
			B = A[keep_index, :]
			Aᵴ = B[:, keep_index]
		
			[ksf_2, eigs_2, num_of_eigs, v2] = self.get_kernel_sampling_feature(Aᵴ, num_of_eigs)
			
			if len(eigs_1) > len(eigs_2):	
				extra_pad = len(eigs_1) - len(eigs_2)
				eigs_2 = np.pad(eigs_2, (0,extra_pad), 'constant')
				
			Kᵭ = np.linalg.norm(eigs_1 - eigs_2, ord=np.inf)

			if Kᵭ < 0.01: 
				return [Aᵴ, v2]




	def double_center(self, Ψ):
		HΨ = Ψ - np.mean(Ψ, axis=0)								# equivalent to Γ = Ⲏ.dot(Kᵧ).dot(Ⲏ)
		HΨH = (HΨ.T - np.mean(HΨ.T, axis=0)).T
		return HΨH


	def get_kernel_sampling_feature(self, rbk, num_of_eigs=None):
		[U,S,V] = np.linalg.svd(rbk)
		v = V[0,:]	# most dominant eig

		cs = np.cumsum(S)/np.sum(S)
		L1_norm = S/np.sum(S)

		if num_of_eigs is None: num_of_eigs = np.sum(cs < 0.9) + 1
		keepCS = cs[0:num_of_eigs]
		L1_norm = L1_norm[0:num_of_eigs]

		return [keepCS, L1_norm, num_of_eigs, v]

	def merge_eig_vectors(self, v1, v2, A):
		n = len(v2)
		sortArgs = np.flip(np.argsort(np.absolute(v1)))
		override = sortArgs[0:n]
		toZero = sortArgs[n:len(v1)]


		sortArgs2 = np.flip(np.argsort(np.absolute(v2)))
		override2 = sortArgs2[0:n]

		print(sortArgs)
		print(override)
		print(toZero,'\n')
		print(sortArgs2, '\n')

		print(v1)
		v_new = v1.copy()
		v_new[toZero] = 0
		print(v_new)

		print(v_new[override])
		print(v2[override2])

		v_new[override] = v2[override2]
		v_new = np.reshape(v_new, (len(v_new),1))
		
		return v_new
		print(v1[override])
		import pdb; pdb.set_trace()

	def get_sparce_eig_vector(self, A):
		inc = 3
		[keepCS, eigs_1, num_of_eigs, v1] = self.get_kernel_sampling_feature(A)
		#v1 = np.reshape(v1, (len(v1),1))

		[Aᵴ, v2] = self.subsample_matrix(A, num_of_eigs, eigs_1)
		vi = self.merge_eig_vectors(v1,v2, A)
		print(v1.T.dot(A).dot(v1))
		print(vi.T.dot(A).dot(vi))
		import pdb; pdb.set_trace()

A = np.random.randn(100,100)
A = A.dot(A.T)
#A =  np.array([[ 5.95 ,  2.558, -0.045,  5.369,  0.571,  4.871, -1.672,  0.784, -1.175, -2.351],
#				[ 2.558,  9.927, -1.97 ,  1.867,  8.123,  1.998, -3.867, -5.423, -4.889, -3.455],
#				[-0.045, -1.97 , 11.314,  3.922, -3.707,  2.685,  5.36 , -4.995, -0.136,  1.396],
#				[ 5.369,  1.867,  3.922,  8.214,  0.626,  3.814, -0.26 , -2.162,  1.232, -1.246],
#				[ 0.571,  8.123, -3.707,  0.626, 12.372, -0.182, -2.772, -1.065, -2.313,  0.384],
#				[ 4.871,  1.998,  2.685,  3.814, -0.182,  6.045,  0.541,  0.586, -3.12 , -1.502],
#				[-1.672, -3.867,  5.36 , -0.26 , -2.772,  0.541,  4.191,  0.548,  0.72 ,  2.633],
#				[ 0.784, -5.423, -4.995, -2.162, -1.065,  0.586,  0.548, 10.719,  2.188,  2.767],
#				[-1.175, -4.889, -0.136,  1.232, -2.313, -3.12 ,  0.72 ,  2.188,  5.608,  2.17 ],
#				[-2.351, -3.455,  1.396, -1.246,  0.384, -1.502,  2.633,  2.767,  2.17 ,  3.465]])
sp = sparse_pca()
sp.get_sparce_eig_vector(A)
