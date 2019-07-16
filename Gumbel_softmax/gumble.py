#!/usr/bin/env python

import numpy as np
import sys

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)



p = np.array([0.6,0.2,0.19,0.01,0])
τ = 0.1
V = np.empty((0, 5))

for m in range(10):
	u = np.random.rand(1,5)
	G = -np.log(-np.log(u))

	c = np.exp((np.log(p) + G)/τ)
	C = c/np.sum(c)
	V = np.vstack((V, C))

print(V)
print(np.mean(V, axis=0))
