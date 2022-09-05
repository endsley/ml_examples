#!/usr/bin/env python

#	This examples shows how we can approximate the eigenvectors of a kernel matrix by combining RFF and nystrom

#	By using RFF, we can get the feature map as computed by RFF directly. This allows us to obtain the eigenfunction in RKHS via
#	1. Given $\psi$ as the feature map, compute the integral operator
#	$$ T_n = \frac{1}{n} \sum_{i=1}^n \psi(x) \psi(x)^T $$
#	2. Find the eigenfunctions
#	$$ [\Phi, \Sigma] = eig(T_n) $$
#	$$ \Sigma = diag(\sigma_1, \sigma_2, ..., ) $$
#	$$ \Phi = [\phi_1, \phi_2, ...]$$
#	3. We next use the eigenfunctions $\Phi$ to approximate the eigenvector of the kernel matrix
#	The key insight is that given an approximation of the eigenfunction $\phi_i$, the corresponding eigenvector $u_i$ of the kernel matrix K is 
#	$$u_i = \frac{1}{\sqrt{\sigma_i}} \Psi_n \phi_i$$
#	$$\Psi_n = \begin{bmatrix} \psi(x_1)^T \\ \psi(x_2)^T \\ ... \\ \psi(x_n) \end{bmatrix} $$
#	$$U = \Psi_n \Phi^T \Sigma$$


import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from sklearn.utils import shuffle
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from tools import *



#	Initialize all the setting
X = csv_load('../dataset/wine.csv', shuffle_samples=True)
q = 30				# size of submatrix A
n = X.shape[0]		# number of total samples
γ = get_rbf_γ(X)	# γ used for the gaussian kerenl

#	Use rff nystrom to approximate K eigenvectors
rbf_feature = RBFSampler(gamma=γ, random_state=1, n_components=100)
Ψ = rbf_feature.fit_transform(X)
Tn = Ψ.T.dot(Ψ)								# unscaled integral operator

#	Note that the approximation is running an eigendecomposition on 100x100 matrix instead of 178 and 178
#	Obviously, the more samples we use the more accurate it would be.
[σs,V] = np.linalg.eig(Tn)					
Σ = np.diag(1/np.sqrt(σs[0:10]))
Φ = V[:, 0:10]
Ū = Ψ.dot(Φ).dot(Σ)		# Note: U bar above implies approximating U actual

#	Use the kernel itself to get the actual eigenvectors
K = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ)
[Λ, U] = np.linalg.eig(K)	# compute the "actual" eigenvectors on a 178 x 178 matrix

#	Print a portion the two eigenvectors at two locations
print_two_matrices_side_by_side(U[0:20, 0:3], Ū[0:20, 0:3], title1='Actual eigenvectors', title2='Approximated Eigenvectors')
print_two_matrices_side_by_side(U[80:100, 4:8], Ū[80:100, 4:8], title1='Actual eigenvectors', title2='Approximated Eigenvectors')


avg_error = np.sum(np.absolute(U[:,0:10] - Ū))/(n*10)
jupyter_print('The average absolute error with RFF Nystrom of each element is %f'% avg_error)












