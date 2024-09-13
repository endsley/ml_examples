#!/usr/bin/env python

import numpy as np
from numpy import array

X = array([	[2,1],
			[3,2],
			[1,0],
			[8,3]])

# default a is not in column format
for xᵢ in X: print(xᵢ)	

# ensure column format by reshaping
for xᵢ in X: print(xᵢ.reshape((2,1)))	

#	If you want to perform
#	$$\sum_{i=1}^4 x_i = x_1 + x_2 + x_3 + x_4$$
y = np.array([[0],[0]])
for xᵢ in X: y = y + xᵢ.reshape((2,1)) 
print(y)

#	Or you can just do
print(np.sum(X, axis=0))
