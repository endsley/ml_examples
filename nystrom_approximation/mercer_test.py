#!/usr/bin/env python

import numpy as np
from tools import *
from sklearn.metrics.pairwise import *


#	This code test the Mercer's Theorem
#	Given
#	1. $K$ as the kernel matrix for $n$ samples.
#	2. $\{e_1, e_2, ..., e_m \}$ as its eigenfunctions 
#	2. $\{v_1, v_2, ...,  \}$ as its eigenvectors
#	3. $\{\sigma_1, \sigma_2, ..., \sigma_m \}$ as its eigenvalues 
#	4. $X \subset \mathbb{R}^n$
#	5. $k: \mathcal{X} \times \mathcal{X}$ be a symmetric continuous kernel function.
#	then Mercer's Thm states that
#
#	$$k(x, y) = \sum_{i=1}^\infty e_i(x) e_i(y)$$
#       $$= \begin{bmatrix} e_1(x) & e_2(x) & ... & e_m(x) & \end{bmatrix} \begin{bmatrix} e_1(x)  \\ e_2(x) \\ ... \\ e_m(x) \\ \end{bmatrix}      $$
#	
#	Implying that using eigenfunction as the basis gives a much smaller set of basis functions as the feature map
#	Here, the eigenfunction is defined as 
#	$$e_i = \frac{1}{\sqrt{\sigma_i}} \Phi^{\top} v_i$$
#
#	Therefore, using the eigenfunction as feature maps, we have
#	$$\Phi = \begin{bmatrix} e_1(x) & e_2(x) & ... \end{bmatrix} = \begin{bmatrix} \Phi e_1 & \Phi e_2 & ... \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{\sigma_1}} \Phi \Phi^{\top} v_1 &  \frac{1}{\sqrt{\sigma_2}} \Phi \Phi^{\top} v_2 &  & ... \end{bmatrix}  = \begin{bmatrix} \frac{1}{\sqrt{\sigma_1}} K v_1 &  \frac{1}{\sqrt{\sigma_2}} K v_2 &  & ... \end{bmatrix} = K \begin{bmatrix} v_1 & v_2 & ... \end{bmatrix} \Sigma = K V \Sigma.$$
#
#	Where 
#	$$ \Sigma = \begin{bmatrix} \frac{1}{\sqrt{\sigma_1}} & 0 & 0 & ... \\  0 & \frac{1}{\sqrt{\sigma_2}} & 0 & ...  \\  ... & ... & ... \end{bmatrix}$$
#
#	<div class="alert alert-block alert-info">
#	<b>Tip:</b> 
#	In this experiment, we are going to 
#	a. generate 10 random samples 
#	b. From these samples, we will directly compute the kernel matrix $K$                                                                           
#	c. After $K$, we are going to use mercer's theorem to generate $\Phi$ with the eigenfunctions                                                   
#	d. If Mercer is correct, then the feature map generated using the eigenfunctions $\Phi$ should give us the condition that $\Phi \Phi^{\top} = K$
#	</div>
	

# Generate kernel matrix
γ = 0.5
X = np.random.randn(10,2)
K = rbf_kernel(X, gamma=γ)

# Generate feature maps via the eigenfunctinos
[D,V] = eigh_sort(K)
Σ = np.diag(1/np.sqrt(D[0:9]))
V = V[:, 0:9]

Φ = K.dot(V).dot(Σ)
K2 = Φ.dot(Φ.T)



#	Remember that since this is a Gaussian Kernel, the feature map should be $\Phi \in \mathbb{R}^{n \times \infty}$, 
#	but through Mercer's theorem, we are apply to obtain $\Phi \in \mathbb{R}^{n \times 9}$. 
#	This is much smaller and easier to deal with. 

#	Lastly, when we print out the kernel matrix of $K$ and $K_2$, notice
#	that they are approximately the same. 



print(K, '\n') 
print(K2)

