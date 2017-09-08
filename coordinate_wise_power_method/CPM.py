#!/usr/bin/env python
#	Assume symmetric matrix

import numpy as np

np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)


def CPM(A):		#	A is a symmetric np array
	n = A.shape[0]	
	x = np.random.randn(n,1)
	x = x/np.linalg.norm(x)
	z = A.dot(x)
	c = np.absolute(z - x)
	loop = True


	while loop: 
		i = np.argmax(c)
	
		y = np.copy(x)
		z_orig = np.copy(z)
		y[i] = (z_orig/(x.T.dot(z_orig)))[i]

		xi = x[i]
		
		z = z_orig + (A[:, np.argmax(c)]*(y[i] - x[i])).reshape(n,1)

		yn = np.linalg.norm(y)
		z = z/yn
		x = y/yn

		y[i] = xi		# turn y into original x
	
		c = np.absolute(x - z/(y.T.dot(z_orig)))

		if(np.sum(np.absolute(x - y)) < 0.00000001): 
			#print A.dot(x)
			print A.dot(x)/x
			print '--------'
			break;

	return x

if __name__ == "__main__":
	A = np.random.randn(10,10)
	A = A.dot(A.T)
	print CPM(A) , '\n'

	[S,V] = np.linalg.eig(A)
	print V, '\n\n'
	print S 
	import pdb; pdb.set_trace()
