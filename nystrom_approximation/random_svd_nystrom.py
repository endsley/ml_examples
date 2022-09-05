#!/usr/bin/env python

#	This examples shows how we can approximate the eigenvectors of a kernel matrix by combining random SVD and nystrom

#	**Method**
#	1. We first subsample p columns, within these p columns, we pick a smaller q columns (p >> q) and use the q columns as L for nystrom
#	2. We find the eigenvector from the q columns to approximate the eigenvectors for p x p matrix as V1
#	3. We next use V1 as a projection matrix for random svd to refine V1 into a better version V2
#	4. We then use V2 (better approximated) again to approximate the eigenvector of the entire kernel matrix K
#
#	Nystrom eigenvector as Q 	-> 	random svd refine the eigenvectors 	-> expand it to the complete Kernel matrix 
#	nystrom expansion			-> 	svd refinement  					-> nystrom expansion


import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from sklearn.utils import shuffle
from sklearn.kernel_approximation import Nystroem
from tools import *

#	Initialize all the setting
X = csv_load('../dataset/wine.csv', shuffle_samples=True)
p = 145				
q = 30				
n = X.shape[0]		# number of total samples
γ = get_rbf_γ(X)	# γ used for the gaussian kerenl


#	Initialize subsamples
Xa = X[0:q, :]	
Xb = X[0:p, :]
sampledK = sklearn.metrics.pairwise.rbf_kernel(X, Y=Xb, gamma=γ)

#	Compute the true kernel from p samples
K = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ)
Kp = sampledK[0:p, 0:p]
[Λ, U] = np.linalg.eig(Kp)	# compute the "actual" eigenvectors


#	**Step 1**
#	Use Nystrom to approximate the initial V1
L = sampledK[0:p, 0:q]
A = L[0:q,:]
[σs,V] = np.linalg.eig(A)
V = V[:,0:10] # only keeping the largest eigenvectors
Σ = np.diag(1/(σs[0:10]))
V1 = L.dot(V).dot(Σ)

#	The result of step 1 give us a bad approximation	
jupyter_print('We used 30 samples to approximate eigenvector of 60 samples (Note: This approximation is not supposed to be good')
print_two_matrices_side_by_side(U[0:15, 0:4], V1[0:15, 0:4], title1='Actual eigenvectors', title2='Approximated Eigenvectors')
avg_error = mean_absolute_error(U[:,0:10], V1, (p*10))
jupyter_print('The average absolute initial error with Nystrom of each element is %f\n\n'% avg_error)

#	**Step 2**
# Use qr to orthogonalize V1 as Q and shrink 	
A2 = sampledK[0:p,0:p]
[Q,R] = np.linalg.qr(V1)		# note that qr here ran on a small matrix
M = Q.T.dot(A2)
[Ư, Σ2, Vᵀ] = np.linalg.svd(M)	# note that the svd here also ran on a small matrix
V2 = Q.dot(Ư)

jupyter_print('We used random SVD to refine the original approximate, this should be better')
print_two_matrices_side_by_side(U[0:15, 0:4], V2[0:15, 0:4], title1='Actual eigenvectors', title2='Approximated Eigenvectors')
avg_error = mean_absolute_error(U[:,0:10], V2, (p*10))
jupyter_print('Notice that the average absolute error after random svd of each element is %f'% avg_error)


jupyter_print('Next, notice that the eigenvalues from random svd and the true eigenvalues are the same')
print('Actual eigenvalues 1st row / Approximated eigenvalues 2nd row')
print(Λ[0:10])
print(Σ2[0:10], '\n\n')

#	**Step 3**
#	Use the result from random SVD as basis of nystrom for the full kernel matrix
Σ3 = np.diag(1/Σ2)
Ꮭ = sampledK 
Ū = Ꮭ.dot(V2).dot(Σ3)

[Λᴋ, Uᴋ] = np.linalg.eig(K)	# compute the "actual" eigenvectors
jupyter_print('We now obtain the eigenvector approximation and compare it to the true eigenvectors')
print_two_matrices_side_by_side(Uᴋ[0:15, 0:4], Ū[0:15, 0:4], title1='Actual eigenvectors', title2='Approximated Eigenvectors')
avg_error = mean_absolute_error(Uᴋ[:,0:10], Ū, (n*10))
jupyter_print('Notice that the average absolute error after random svd of each element is %f'% avg_error)


