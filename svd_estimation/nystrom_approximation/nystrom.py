#!/usr/bin/env python

#	This code test the Nystrom method
#	Given a symmetrick kernel matrix $K \in \mathbb{R}^{n \times n}$ and a submatrix L
#	$$K = \begin{bmatrix} A & B^T \\ B & C \end{bmatrix}, \quad L = \begin{bmatrix} A \\ B \end{bmatrix}$$
#	We define the eigenvalues of matrix $A \in \mathbb{R}^{q \times q}$ (where $q << n$) as 
#	$$\sigma(A) = \{\sigma_1, \sigma_2, ..., \sigma_q\}$$
#	We define the eigenvectors of matrix $A$ as 
#	$$V(A) = \{v_1, v_2, ..., v_q\}$$

#	The Nystrom method allows us to compute 
#	1. The Entire $K$ matrix using $L$
#	2. The eigenvectors of $K$ using the eigenvectors of $A$
#	3. The inverse of $K$


#	The Nystrom method explained
#	1. We randomly sample q samples and use that to form the top left matrix A
#	2. We find $\sigma(A)$ and $V(A)$ 
#	3. We use the r most dominant eigenvectors of A to approximate the eigenfunctions $\phi$
#		If we define the feature map 
#		$$\Psi_q = \begin{bmatrix} \psi(x_1)^T \\ \psi(x_2)^T \\ .. \\ \psi(x_q)^T \end{bmatrix}$$
#		Then the relationship between the eigenfunction $\phi_i$ and the eigenvector $v_i$ is
#		$$\phi_i = \frac{1}{\sqrt{\sigma_i}} \Psi_q^T v_i$$
#	4. Once we approximated the eigenfunction, we can use it to approximate $K$ via the following derivation
#	$$K = \Phi \Phi^T$$
#	$$K = \begin{bmatrix}  \phi_1(x_1) & \phi_2(x_1) & .. & \phi_r(x_1)\\  \phi_1(x_2) & \phi_2(x_2) & .. & \phi_r(x_2)\\ ... & ... & ... & ...\\ \phi_1(x_n) & \phi_2(x_n) & .. & \phi_r(x_n) \end{bmatrix} \begin{bmatrix}  \phi_1(x_1) & \phi_2(x_1) & .. & \phi_r(x_1)\\  \phi_1(x_2) & \phi_2(x_2) & .. & \phi_r(x_2)\\ ... & ... & ... & ...\\ \phi_1(x_n) & \phi_2(x_n) & .. & \phi_r(x_n) \end{bmatrix}^T $$
#	$$K = \begin{bmatrix} \frac{1}{\sqrt{\sigma_1}} \Psi_n \Psi_q^T v_1  & \frac{2}{\sqrt{\sigma_2}} \Psi_n \Psi_q^T v_2 & ... & \frac{1}{\sqrt{\sigma_r}} \Psi_n \Psi_q^T v_r\end{bmatrix} \begin{bmatrix}  \phi_1(x_1) & \phi_2(x_1) & .. & \phi_r(x_1)\\  \phi_1(x_2) & \phi_2(x_2) & .. & \phi_r(x_2)\\ ... & ... & ... & ...\\ \phi_1(x_n) & \phi_2(x_n) & .. & \phi_r(x_n) \end{bmatrix}^T $$
#	$$K =  (\Psi_n \Psi_q^T V \Sigma) (\Psi_n \Psi_q^T V \Sigma)^T$$
#	where
#	$$\Sigma = \begin{bmatrix} \frac{1}{\sqrt{\sigma_1}} & 0 & 0 & ... \\ 0 &  \frac{1}{\sqrt{\sigma_2}} & 0 & ... \\ 0 &  0 & \frac{1}{\sqrt{\sigma_2}} & ... \\ ... &  ... & ... & ... \end{bmatrix} $$
#	Therefore we have
#	$$K =  (LV\Sigma) (\Sigma V^TL^T)$$
#	$$K =  \Phi \Phi^T$$



import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from sklearn.utils import shuffle
from sklearn.kernel_approximation import Nystroem
from tools import *

#	Initialize all the setting
X = csv_load('../../dataset/wine.csv', shuffle_samples=True)
q = 30				# size of submatrix A
n = X.shape[0]		# number of total samples
γ = get_rbf_γ(X)	# γ used for the gaussian kerenl


#	Use Nystrom to approximate the kernel
Xa = X[0:q, :]	# A will come from Xa
L = sklearn.metrics.pairwise.rbf_kernel(X, Y=Xa, gamma=γ)
A = L[0:q,:]
[σs,V] = np.linalg.eig(A)
V = V[:,0:10] # only keeping the largest eigenvectors
Σ = np.diag(1/(np.sqrt(σs[0:10])))
Φ = L.dot(V).dot(Σ)
ǩ = Φ.dot(Φ.T)


#	Use SKlearn to approximate the kernel
feature_map_nystroem = Nystroem(gamma=γ, random_state=1, n_components=q)
Φ2 = feature_map_nystroem.fit_transform(X)
K2 = Φ2.dot(Φ2.T)

#	Compute the actual kernel
K = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ)


#	Display the results
#	Notice that even though we only used 30 samples to approximate the entire 178 sample, 
#	we still got really good approximation
a= 100; b= 107
print_two_matrices_side_by_side(K[a:b, a:b], ǩ[a:b, a:b], title1='Actual Kernel', title2='Nystrom Kernel')
print_two_matrices_side_by_side(K[a:b, a:b], ǩ[a:b, a:b], title1='Actual Kernel', title2='Sklearn Nystrom Kernel')

avg_error = np.sum(np.absolute(K - ǩ))/(n*n)
avg_error2 = np.sum(np.absolute(K - K2))/(n*n)

jupyter_print('The average absolute error with Nystrom of each element is %f'% avg_error)
jupyter_print('The average absolute error with Sklearn Nystrom of each element is %f'% avg_error2)


