#!/usr/bin/env python


#	This code tests the idea of random SVD. 
#	Random SVD is a way of performing fasters SVD by randomly 
#	project the data down to a lower dimension. 

#	**Method**
#	Suppose we have Q matrix with orthonormal columns with the properties : 
#	$$ K \approx QQ^* K$$
#	that is, ideally $QQ^* = I$ 

#	The idea of random svd, is that if we know Q, we can perform svd on 
#	the significantly smaller matrix of $Q^*K$ with the following derivation
#	$$ K \approx Q (Q^* K)$$
#	$$ K \approx Q (M)$$
#	$$ K \approx Q (UΣV^T)$$

#	Note that $M = UΣV^T$

#	Notice that while $K \in \mathbb{R}^{n \times n}$ the dimension of $Q^*K$ is $\mathbb{R}^{n \times q}$ where $q$ is very small.
#	(Note: q is normally just set to values between 10 to 20)
#	Since $q$ is small the svd of $(Q^* K)$ is performed on a $q \times q$ matrix.
#	This can be done very cheap. 

#	Once you have performed SVD on $(Q^* K) = UΣV^T$, the eigenvector the K is
#	$$eigV(K) \approx QU$$

#	The key question is "How do we get Q?", this is normally done by first generating
#	a matrix $\Omega \in \mathbb{R}^{n \times q}$ filled by normal Gaussian distribution
#	 We project the K onto $\Omega$ 
#	$$Y = K\Omega$$

#	Note that $Y$ only has $q$ columns and a QR decomposition can be very cheaply conducted 
#	when there are few columns. 
#	$$ [Q,R] = qr(Y)$$

#	This allows us to get the Q matrix where $QQ^* = I$


import numpy as np
import sklearn
import sklearn.metrics
from sklearn.metrics import mean_absolute_error
from tools import *


X = csv_load('../dataset/wine.csv', shuffle_samples=True)
n = X.shape[0]		# number of total samples
γ = get_rbf_γ(X)	# γ used for the gaussian kerenl

K = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ)
Ω = np.random.randn(n,10)
Y = K.dot(Ω) #note that Y is a tall and smaller matrix compared to K, therefore QR is cheaper
print(Y.shape)

[Q,R] = np.linalg.qr(Y)
M = Q.T.dot(K)
[Ū, Σ, Vᵀ] = np.linalg.svd(M)
U = Q.dot(Ū)		
print(Ū.shape)						#notice that Ū is very small
print(U.shape)						#notice that U is large

#	Compute the actual eigenvectors of K
[Λ,V] = np.linalg.eig(K)

#	Visually compare the two eigenvectors
MAE = mean_absolute_error(V[:,0:4], U[:,0:4])
print_two_matrices_side_by_side(U[30:50,0:3], V[30:50,0:3], title1='Approximate eigV', title2='Actual eigV', auto_print=True)
jupyter_print('The mean absolute error is : %.3f'% MAE)
