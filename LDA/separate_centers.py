#!/usr/bin/env python

import numpy as np
from numpy import mean, reshape, array
from numpy import vstack, hstack
from matplotlib import pyplot as plt


X0 = array([[-2,1], [0,3], [1,4], [1,5], [2,5]]) 
X1 = array([[0,0], [1,1], [2,1], [3,2], [4,3]]) 

# 	Finding S1
v = reshape(mean(X0, axis=0) - mean(X1, axis=0), (2,1))
S1 = v.dot(v.T)

#	Finding the vector ṽւ that maximize center distance
[D,V] = np.linalg.eigh(S1)
ṽւ = reshape(V[:,1], (2,1))
linePoints = vstack((5*ṽւ.T, -5*ṽւ.T))

#	Project the data onto the LDA line
X0ᴾ = (X0 ).dot(ṽւ).dot(ṽւ.T)		# data after LDA projection
X1ᴾ = (X1 ).dot(ṽւ).dot(ṽւ.T)		# data after LDA projection

plt.figure(figsize = (5, 5))
plt.plot(linePoints[:,0], linePoints[:,1], color='blue')
plt.scatter(X0[:,0], X0[:,1], color='red', s=16)
plt.scatter(X0ᴾ[:,0], X0ᴾ[:,1], color='red', s=16)
plt.scatter(X1[:,0], X1[:,1], color='green', s=16)
plt.scatter(X1ᴾ[:,0], X1ᴾ[:,1], color='green', s=16)
plt.xlim([-3,6])
plt.ylim([-3,6])
plt.title('Original Points vs Projected Points')
plt.tight_layout()
plt.show()

