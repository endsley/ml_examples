#!/usr/bin/env python

from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from numpy import exp as e
from numpy import zeros, array, reshape

#	Write my own kernel
X = array([[1,2],[0,1],[1,0]])

def k(xi, xj):
	γ = 1
	d = (xi - xj).T.dot((xi - xj))
	return e(-γ*d)

K = zeros((3,3))
for i, xi in enumerate(X):
	for j, xj in enumerate(X):
		xi = reshape(xi, (2,1))
		xj = reshape(xj, (2,1))
		K[i,j] = k(xi, xj)

print(K)

#	Use sklearn to generate the kernel
#	Notice it is the same as calculated by the equation
K = rbf_kernel(X, gamma=1)
print(K)
