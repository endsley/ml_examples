#!/usr/bin/env python

import numpy as np
import sys
from time import perf_counter
from scipy.linalg import hadamard


def fwht(a):		
	"""In-place Fast Walsh–Hadamard Transform of array a."""
	a = a.tolist()
	h = 1
	while h < len(a):
		for i in range(0, len(a), h * 2):
			for j in range(i, i + h):
				x = a[j]
				y = a[j + h]
				a[j] = x + y
				a[j + h] = x - y
		h *= 2

	a = (1/np.sqrt(2))*np.array(a)
	return a
   
if __name__ == "__main__":
	np.set_printoptions(precision=2)
	np.set_printoptions(threshold=30)
	np.set_printoptions(linewidth=300)
	np.set_printoptions(suppress=True)
	np.set_printoptions(threshold=sys.maxsize)


	#x = np.random.random(1024**2)
	l = 8
	x = np.random.random(l)

	t2 = perf_counter()
	y2 = fwht(x)		# input is 1 dimensional array
	t2 = perf_counter() - t2
	print(t2, ' : ', y2)

	t3 = perf_counter()
	H = hadamard(l)
	y3 = (1/np.sqrt(2))*H.dot(np.reshape(x,(l,1)))
	t3 = perf_counter() - t3
	print(t3, ' : ', y3.T)
