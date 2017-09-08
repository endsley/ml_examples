#!/usr/bin/python

import numpy as np

def eig_sorted(X):
	D,V = np.linalg.eig(X)	
	lastV = None
	sort_needed = False
	for m in D:
		if m > lastV and lastV != None:
			sort_needed = True
			#print 'Sort needed : \t' , m, lastV
		lastV = m
	
	if sort_needed:
		idx = D.argsort()[::-1]   
		D = D[idx]
		V = V[:,idx]	

	return [V,D] 

if __name__ == '__main__':
	from random_matrix import *
	M = random_matrix(2, 5)
	[V,D] = eig_sorted(M)
	
	print 'V : \n' , V , '\n'
	print 'D : \n' , D , '\n'
	print 'M : \n' , M , '\n'

	print M
	#print V.dot(D).dot(V.T)
	print '\n'
	print V.dot(np.diag(D)).dot(V.T)

#	print '\n'
#	print V
#	print '\n'
#	print D
	
