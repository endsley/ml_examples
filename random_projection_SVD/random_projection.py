#!/usr/bin/env python


#	This code tests the idea of random SVD. 
#	Random SVD is a way of performing fasters SVD by randomly 
#	project the data down to a lower dimension. 

#	Method
#	Suppose we have Q matrix with orthonormal columns with the properties : 
#	$$ A \approx QQ^* A$$
#	that is, ideally $QQ^* = I$ 

#	The idea of random svd, is that if we know Q, we can perform svd on 
#	the significantly smaller matrix of $Q^*A$ with the following derivation
#	$$ A \approx Q (Q^* A)$$
#	$$ A \approx Q (UΣV^T)$$

#	Notice that while $A \in \mathbb{R}^{n \times n}$ the dimension of $Q^*A$ 
#	is $\mathbb{R}^{n \times q}$ where $q$ is very small.
#	(Note: q is normally just set to values between 10 to 20)
#	Since $q$ is small the svd of $(Q^* A)$ is performed on a $q \times q$ matrix.
#	This can be done very cheap. 

#	Once you have performed SVD on $(Q^* A) = UΣV^T$, the eigenvector the A is
#	$$eigV(A) \approx QU$$

#	The key question is "How do we get Q?", this is normally done by first generating
#	a matrix $\Omega \in \mathbb{R}^{n \times q} filled by normal Gaussian distribution
#	 We project the A onto $\Omega$ 
#	$$Y = A\Omega$$

#	Note that $Y$ only has $q$ columns and a QR decomposition can be very cheaply conducted 
#	when there are few columns. 
#	$$ [Q,R] = qr(Y)$$

#	This allows us to get the Q matrix where $QQ^* = I$

