#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize

#	Given
#	$$A = \begin{bmatrix} 2 & 3 \\ 3 & 5 \end{bmatrix}, y = \begin{bmatrix} 2 \\ 1 \end{bmatrix} $$
#	Minimize
#	$$\min_x \; x^{\top} A x - y^{\top} x + ||x||_2^2$$


A =  np.array([[2, 3],[3, 5]])
y = np.array([[2],[1]])

#	define the function to minimize
def f(x): return x.T.dot(A).dot(x) - y.T.dot(x) + x.T.dot(x)

x0 = np.array([3,2])	# x0 is the starting point
result = minimize(f, x0)
print(result)

