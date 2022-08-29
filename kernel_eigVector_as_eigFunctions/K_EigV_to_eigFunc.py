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

#	Given $\phi_i$ as the feature map and $n$ as the number of samples, we define the integral operator $T_n$ as 
#	$$T_n = \frac{1}{n}\sum_i \phi_i(x_i) \phi_i^T(x_i) $$

#	In this experiment, we will use  the feature map of a polynomial kernel 
#	$k(x,y) = (x^Ty + 1)^2$
#	We will assume that each sample has two dimensions, therefore the feature map is finite 
#	$\phi(x) = [x_1^2,  \sqrt{2} x_1x_2, \sqrt{2} x_1, \sqrt{2} x_2, x_2^2, 1]$
#	By using this equation, we can compute Tn, the approximate integral operator. 


def polyK(X):	# calculate the polynomial kernel matrix
	K = np.zeros((len(X), len(X)))
	for i, x in enumerate(X):
		for j, y in enumerate(X):
			xi = x.reshape(len(x), 1)
			yi = y.reshape(len(y), 1)
			K[i,j] = (xi.T.dot(yi) + 1)**2
	return K

def Φ(X):		# this function takes x and calculate its feature map
	fm = np.empty((6, 0))
	c = np.sqrt(2)
	for x in X:
		# feature map of polynomial [x₁ᒾ,  ᴄ x₁x₂, ᴄ x₁, c x₂, x₂ᒾ, 1]
		φ = np.array([[x[0]*x[0], c*x[0]*x[1], c*x[0], c*x[1], x[1]*x[1], 1]]).T
		fm = np.hstack((fm, φ))
	return fm.T


if __name__ == "__main__":
	jprint = jupyter_print


	X = np.array([[1,1], [2,2], [3,3]])
	n = X.shape[0]
	#	
	Φ = Φ(X)						# feature map
	K = polyK(X)					# X to kernel matrix
	#
	Tn = (1/n)*Φ.T.dot(Φ)			# The approximate integral operator
	#
	[λ, V] = eig(K)				# eigenvalue/vector of the kernel matrix
	[σ, U] = eig(Tn)				# eigenvalue/function of the integral operator



	#
	eK = pretty_np_array(λ, front_tab='', title='Eig Values of K', auto_print=False)
	eQ = pretty_np_array(σ, front_tab='', title='Eig Values of Tn', auto_print=False)
	block_two_string_concatenate(eK, eQ, spacing='\t', add_titles=[], auto_print=True)
	
	#Notice that the eigenvectors of K is not the same as $T_n$!!!
	#However if we multiply the eigenvalues by n, they become the same.
#
	nσ = n*σ[0:3]
	σ_text = pretty_np_array(nσ, front_tab='', title='Eigenvalues after multiplied by n', auto_print=False, end_space=" = 3 * ") 
	eQn = pretty_np_array(σ[0:3], front_tab='', title='Original eigenvalues', auto_print=False)
	display1 = block_two_string_concatenate(σ_text, eQn, spacing=' ', add_titles=['Eigs'], auto_print=False)
	jprint(display1)

#	Therefore, we conclude the relationship between eigenvalues of kernel matrix and $T_n$ is
#	$$\mathbf{\sigma(K) = n*\sigma(T_n)}$$

#	We next verify the eigenvector eigenfunction relationship.
#	We first show the eigenvectors of K and eigenfunctions from Tn

	eigK = pretty_np_array(V, front_tab='', title='Eig of K', auto_print=False)
	eigQ = pretty_np_array(U, front_tab='', title='Eig of Tn', auto_print=False)
	block_two_string_concatenate(eigK, eigQ, spacing='\t', add_titles=[], auto_print=True)
#

#	Given $f_i$ as the $i_{th}$ eigenfunction associated with the eigenvector of $K$ denoted as $v_i$.
#	Given $\sigma_i$ as the eigenvalues of the integral operator (not from the kernel)
#	Note that the relationship between the eigenvectors of K and eigenfunction is
#	$$f_i = \frac{1}{\sqrt{n \sigma_i}} \Phi^T v_i$$

#	where
#	$$\Phi^T = \begin{bmatrix} \phi(x_1) &  \phi(x_2) & ... & \phi(x_n) \end{bmatrix}$$

#	Notice that our eigenvector eigenfunction relationship is slightly different from the equation proposed by 
#	Rosasco, and that is because he has defined the kernel matrix differently. 

	σ3 = σ[0:3]
	eigFun_from__eigV_of_K = (1/np.sqrt(n*σ3))*((Φ.T.dot(V)))
	eigFun = U[:,0:3]
#
	jprint('Notice how the eigenfunction computed from the eigenvectors is identical to the actual eigenfunctions !!!')
	eigFun_from__eigV_of_K = pretty_np_array(eigFun_from__eigV_of_K, front_tab='', title='Computed eig Function', auto_print=False)
	eigFun = pretty_np_array(eigFun, front_tab='', title='True Eigen Function', auto_print=False)
	block_two_string_concatenate(eigFun_from__eigV_of_K, eigFun, spacing='\t', add_titles=[], auto_print=True)

#	Alternatively, if we let $\lambda_i$ be the eigenvalues from the kernel matrix K, then the relationship would be
#	$$f_i = \frac{1}{\sqrt{\lambda_i}} \Phi^T v_i$$

	eigFun_from__eigV_of_K = (1/np.sqrt(λ))*((Φ.T.dot(V)))
	eigFun = U[:,0:3]
#
	jprint('Notice how the eigenfunction computed from the eigenvectors is identical to the actual eigenfunctions !!!')
	eigFun_from__eigV_of_K = pretty_np_array(eigFun_from__eigV_of_K, front_tab='', title='Computed eig Function', auto_print=False)
	eigFun = pretty_np_array(eigFun, front_tab='', title='True Eigen Function', auto_print=False)
	block_two_string_concatenate(eigFun_from__eigV_of_K, eigFun, spacing='\t', add_titles=[], auto_print=True)



#	Next, we can get the inverse relationship where we go from the eigenfunction back to the eigenvector. 
#	We use the following derivation
#	$$f_i = \frac{1}{\sqrt{\lambda_i}} \Phi^T v_i$$
#	$$\Phi f_i = \frac{1}{\sqrt{\lambda_i}} \Phi \Phi^T v_i$$
#	$$\Phi f_i = \frac{1}{\sqrt{\lambda_i}} K v_i$$
#	$$\Phi f_i = \frac{\lambda_i}{\sqrt{\lambda_i}} v_i$$
#	$$\Phi f_i = \sqrt{\lambda_i} v_i$$
#	$$\frac{1}{\sqrt{\lambda_i}}\Phi f_i = v_i$$
#	$$\frac{1}{\sqrt{\lambda_i}}\Phi \begin{bmatrix} f_1 & f_2 & .. \end{bmatrix} = \begin{bmatrix} v_1 & v_2 & .. \end{bmatrix}$$
