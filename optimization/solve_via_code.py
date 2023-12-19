#!/usr/bin/env python

import numpy as np

v = 10**10
for i in range(3000):
	x1 = np.random.rand()
	x2 = 1 - x1

	u = 2*x1**2 + x2**2
	if u < v:
		v = u
		print(v)
