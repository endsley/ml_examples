#!/usr/bin/env python

import numpy as np
import sys
from time import perf_counter
from scipy.linalg import hadamard


def fwht(x):
	"""In-place Fast Walsh–Hadamard Transform of array a."""
	a = x.T.tolist()

	if len(x.shape) == 1:
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
		a = np.reshape(a,(a.shape[0],1))
		return a

	elif len(x.shape) == 2:
		b = x.T.tolist()

		for a in b: 
			h = 1
			while h < len(a):
				for i in range(0, len(a), h * 2):
					for j in range(i, i + h):
						x = a[j]
						y = a[j + h]
						a[j] = x + y
						a[j + h] = x - y
				h *= 2
		
		b = (1/np.sqrt(2))*np.array(b).T
		return b

   
if __name__ == "__main__":
	np.set_printoptions(precision=2)
	np.set_printoptions(threshold=30)
	np.set_printoptions(linewidth=300)
	np.set_printoptions(suppress=True)
	np.set_printoptions(threshold=sys.maxsize)


	#x = np.random.random(1024**2)
	l = 4
	d = 2
	x = np.random.random((l,d))

	t2 = perf_counter()
	y2 = fwht(x)		# input is 1 dimensional array
	t2 = perf_counter() - t2

	t3 = perf_counter()
	H = hadamard(l)
	y3 = (1/np.sqrt(2))*H.dot(x)
	t3 = perf_counter() - t3

	#print(t2)
	#print(t3)

	print(t2, ' : \n', y2)
	print(t3, ' : \n', y3)
