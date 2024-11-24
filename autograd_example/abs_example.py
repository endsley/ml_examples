#!/usr/bin/env python
# Automatically find the gradient of a function
# Download the package at : https://github.com/HIPS/autograd

import autograd.numpy as np
from autograd import grad


#	Given the function
#	$$f(x) = ||W x||_1$$
#	$$W = \begin{bmatrix} 2 & 3 \\ 1 & 4\end{bmatrix}, \quad x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$
#	The derivative is 
#	$$ f'(x) = W^{\top} Sign(Wx) $$
#	The derivative is from the following
#	$$ f(x) = \left| \left| \begin{bmatrix} 2 & 3 \\ 1 & 4\end{bmatrix}\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \right| \right| = |2x_1 + 3x_2| + |x_1 + 4x_2|$$
#	$$ \frac{df}{dx} = \begin{bmatrix} 2 sign(2x_1 + 3x_2) + sign(x_1 + 4x_2) \\ 3 sign(2x_1 + 3x_2) + 4 sign(x_1 + 4x_2) \end{bmatrix} $$
#	$$ \frac{df}{dx} = \begin{bmatrix} 2 & 1 \\ 3 & 4\end{bmatrix}\begin{bmatrix} sign(2x_1 + 3x_2) \\ sign(x_1 + 4x_2 \end{bmatrix} = W^{\top} sign(Wx) $$


title = np.array([['fᑊ', 'ߜf']])
W = np.array([[2,3],[1,4]])
x = np.random.randn(2,1)

def f(x): 
	return np.linalg.norm(W.dot(x), ord=1)

def ߜf(x):
	return W.T.dot(np.sign(W.dot(x)))

fᑊ = grad(f)       # Obtain its gradient function

for i in range(10):
	x = np.random.randn(2,1)
	print(np.vstack((title, np.hstack((	fᑊ(x), ߜf(x))))), '\n')

