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
X = csv_load('../dataset/wine.csv', shuffle_samples=True)
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


avg_error = mean_absolute_error(K, ǩ, (n*n))
avg_error2 = mean_absolute_error(K, K2, (n*n))

jupyter_print('The average absolute error with Nystrom of each element is %f'% avg_error)
jupyter_print('The average absolute error with Sklearn Nystrom of each element is %f\n\n'% avg_error2)



#	**How to use Nystrom to approximate eigenvectors of a large matrix** 
#	The key insight is that given an approximation of the eigenfunction $\phi_i$, the corresponding eigenvector $u_i$ of the kernel matrix K is 
#	$$u_i = \frac{1}{\sqrt{\sigma_i}} \Psi_n \phi_i$$

#	**Method**
#	1. use the eigenvectors $V$ of matrix A to approximate the eigenfunction
#	where $V = [v_1, v_2, ..., v_q]$, we get an expression for the eigenfunction
#	$$\phi_i = \frac{1}{\sqrt{\sigma_i}} \Psi_q^T v_i$$

#	2. Next, we plug the eigenvector into the previous equation to get the eigenvector
#	$$u_i = \frac{1}{\sqrt{\sigma_i}} \Psi_n (\frac{1}{\sqrt{\sigma_i}} \Psi_q^T v_i)$$
#	$$u_i = \frac{1}{\sigma_i} \Psi_n \Psi_q^T v_i$$
#	$$u_i = \frac{1}{\sigma_i} L v_i$$
#	$$U = L V \begin{bmatrix} \frac{1}{\sigma_1} & 0 & 0 & ... \\ 0 & \frac{1}{\sigma_2} & 0 & ... \\  0 & 0 & \frac{1}{\sigma_3} & 0 & ... \\   \end{bmatrix}$$
#	$\Sigma$ is a diagonal matrix
#	$$\bar{U} = L V \Sigma$$


jupyter_print("We now approximate the eigenvector with only 30 samples")
Σ = np.diag(1/σs[0:10]) 	# notice that Σ here is defined slightly differently
Ū = L.dot(V).dot(Σ)			# approximate eigenvector of the larger matrix
[Λ, U] = np.linalg.eig(K)	# compute the "actual" eigenvectors

jupyter_print("Notice that the approximation is not that great unless you are using a large amount of samples. ")
jupyter_print("For this reason, it makes sense to combine random svd with nystrom to approximate the eigenvectors")
print_two_matrices_side_by_side(U[0:10, 0:3], Ū[0:10, 0:3], title1='Actual eigenvectors', title2='Approximated Eigenvectors')

avg_error = mean_absolute_error(U[:,0:10], Ū[:,0:10], (n*10))
jupyter_print('The average absolute error of each element is %f\n'% avg_error)

jupyter_print("Let's perform the nystrom eigenvector approximation, but with a lot more samples, q=150, instead of just 30 samples")
#	Initialize all the setting
q = 150				# size of submatrix A

#	Use Nystrom to approximate the kernel
Xa = X[0:q, :]	# A will come from Xa
L = sklearn.metrics.pairwise.rbf_kernel(X, Y=Xa, gamma=γ)
A = L[0:q,:]
[σs,V] = np.linalg.eig(A)
V = V[:,0:10] # only keeping the largest eigenvectors

Σ = np.diag(1/σs[0:10]) 	# notice that Σ here is defined slightly differently
Ū = L.dot(V).dot(Σ)			# approximate eigenvector of the larger matrix

jupyter_print("Notice how much more accurate the approximation becomes!!!")
print_two_matrices_side_by_side(U[0:10, 0:3], Ū[0:10, 0:3], title1='Actual eigenvectors', title2='Approximated Eigenvectors')

avg_error = mean_absolute_error(U[:,0:10], Ū[:,0:10], (n*10))
jupyter_print('The average absolute error of each element is %f\n'% avg_error)
