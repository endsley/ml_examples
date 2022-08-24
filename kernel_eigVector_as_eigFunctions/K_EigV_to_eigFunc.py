#!/usr/bin/env python

import numpy as np
import sys
from numpy.linalg import eig
from numpy.linalg import svd
from tools import *

np.set_printoptions(precision=4)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)


#	This code tests Proposition 9 of the paper 
#	ONLEARNING WITHINTEGRALOPERATORS
#	The idea is that we know what the relationship between eigenvectors of the kernel matrix and the eigen functions.

#	Given $\phi_i$ as the eigenfunctions and $m$ as the number of eigenfunctions, we define the integral operator $T_n$ as 
#	$$T_n = \frac{1}{m}\sum_i \phi_i \phi_i^T $$




def polyK(X):	# calculate the polynomial kernel matrix
	K = np.zeros((len(X), len(X)))
	for i, x in enumerate(X):
		for j, y in enumerate(X):
			xi = x.reshape(len(x), 1)
			yi = y.reshape(len(y), 1)
			K[i,j] = (xi.T.dot(yi) + 1)**2
	return K

def Φ(X):
	fm = np.empty((6, 0))
	c = np.sqrt(2)
	for x in X:
		# feature map of polynomial [x₁ᒾ,  ᴄ x₁x₂, ᴄ x₁, c x₂, x₂ᒾ, 1]
		φ = np.array([[x[0]*x[0], c*x[0]*x[1], c*x[0], c*x[1], x[1]*x[1], 1]]).T
		fm = np.hstack((fm, φ))
	return fm.T


if __name__ == "__main__":
	X = np.array([[1,1], [2,2], [3,3]])
	n = X.shape[0]
	#	
	Q = Φ(X)						# feature map
	K = polyK(X)					# X to kernel matrix
	#
	Tn = (1/n)*Q.T.dot(Q)			# The approximate integral operator
	#
	[Dk, Vk] = eig(K)				# eigenvalue/vector of the kernel matrix
	[Dq, Vq] = eig(Tn)				# eigenvalue/function of the integral operator
	#
	eK = pretty_np_array(Dk, front_tab='', title='Eig Values of K', auto_print=False)
	eQ = pretty_np_array(Dq, front_tab='', title='Eig Values of Tn', auto_print=False)
	block_two_string_concatenate(eK, eQ, spacing='\t', add_titles=[], auto_print=True)
	#
	jupyter_print('Notice that the eigenvectors of K is not the same as Tn!!!')
	jupyter_print('However if we multiply the eigenvalues by n, they become the same.\n')
#
#
	Dqn = n*Dq[0:3]
	DQ_text = pretty_np_array(Dqn, front_tab='', title='Eigenvalues after multiplied by n', auto_print=False, end_space=" = 3 * ") 
	eQn = pretty_np_array(Dq[0:3], front_tab='', title='Original eigenvalues', auto_print=False)
	display1 = block_two_string_concatenate(DQ_text, eQn, spacing=' ', add_titles=['Eigs'], auto_print=False)
	jupyter_print(display1)



	jupyter_print('Therefore, we conclude that σ(K) = n*σ(Tn), we next verify the eigenvector eigenfunction relationship.\n')
	jupyter_print('We first show the eigenvectors of K and eigenfunctions from Tn.')
#
	eigK = pretty_np_array(Vk, front_tab='', title='Eig of K', auto_print=False)
	eigQ = pretty_np_array(Vq, front_tab='', title='Eig of Tn', auto_print=False)
	block_two_string_concatenate(eigK, eigQ, spacing='\t', add_titles=[], auto_print=True)
#
	jupyter_print('Note that if we use the equation:')

#	Given $f_i$ as the $i_{th}$ eigenfunction associated with the eigenvector of $K$ denoted as $v_i$.
#	Given $\sigma_i$ as the eigenvalues of the integral operator (not from the kernel)
#	$$f_i = \frac{1}{\sqrt{n \sigma_i}} \Phi^T v_i$$

#	Alternatively, if we let $\lambda_i$ be the eigenvalues from the kernel matrix K, then the relationship would be
#	$$f_i = \frac{1}{\sqrt{\lambda_i}} \Phi^T v_i$$

#	Note that
#	$$\Phi^T = \begin{bmatrix} \phi(x_1) &  \phi(x_2) & ... & \phi(x_n) \end{bmatrix}$$


	Dq3 = Dq[0:3]
	eigFun_from__eigV_of_K = (1/np.sqrt(n*Dq3))*((Q.T.dot(Vk)))
	eigFun = Vq[:,0:3]
#
	jupyter_print('Notice how the eigenfunction computed from the eigenvectors is identical to the actual eigenfunctions !!!')
	eigFun_from__eigV_of_K = pretty_np_array(eigFun_from__eigV_of_K, front_tab='', title='Computed eig Function', auto_print=False)
	eigFun = pretty_np_array(eigFun, front_tab='', title='True Eigen Function', auto_print=False)
	block_two_string_concatenate(eigFun_from__eigV_of_K, eigFun, spacing='\t', add_titles=[], auto_print=True)

