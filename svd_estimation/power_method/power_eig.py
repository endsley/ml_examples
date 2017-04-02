#!/usr/bin/python

import numpy as np
import time 

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

def multiply_until_converge(X, f, accuracy):
	for p in range(1000):
		new_f = X.dot(f)
		new_f = new_f/np.max(new_f)
		if np.linalg.norm(new_f - f) < accuracy*np.linalg.norm(f):
			#print np.linalg.norm(new_f - f)
			print 'iteration : ' , p
			break
		else:
			f = new_f

	f = f/np.linalg.norm(f)
	return f

def power_eig(B, num_eigs, accuracy=0.0001, direction='largest first'): # direction = 'largest first' or 'smallest first'
	if B.shape[0] != B.shape[1]:
		print 'Error : A must be a squared matrix'
		return

	d = B.shape[0]
	eigValues = np.array([])
	eigVects = np.empty((d, 0))
	

	if direction == 'smallest first':
		N = np.abs(np.max(B))
		X = N*np.eye(d) - B
		
		for e in range(num_eigs):
			f = np.random.randn(d,1)
			f = multiply_until_converge(X, f, accuracy)
			ev = f.T.dot(X).dot(f)
			eig_value = N - ev
			X = X - ev*f.dot(f.T)

			eigValues = np.append(eigValues, eig_value)
			eigVects = np.hstack((eigVects, f))
	else:
		for e in range(num_eigs):
			f = np.random.randn(d,1)
			f = multiply_until_converge(B, f, accuracy)
			eig_value = f.T.dot(B).dot(f)
			B = B - eig_value*f.dot(f.T)
	
			eigValues = np.append(eigValues, eig_value)
			eigVects = np.hstack((eigVects, f))

	return [eigVects, eigValues]


if __name__ == "__main__":
	np.set_printoptions(precision=4)
	np.set_printoptions(threshold=np.nan)
	np.set_printoptions(linewidth=300)
	

	
	A = np.random.randn(6,6)
	A = A.dot(A.T)
	[V,D] = eig_sorted(A)

	print 'Truth eigen decomposition : ' 
	print V , '\n\n'
	print D , '\n----------\n'

	print 'Finding Largest 2 eigenvalues and vectors : \n'
	[eigVects, eigValues] = power_eig(A,2, accuracy=0.00001)
	print 'Eigenvectors : \n',  eigVects , '\n\n'
	print 'Eigenvalues : \n', eigValues , '\n----------\n'
	
	
	print 'Finding smallest 2 eigenvalues and vectors : \n'
	[eigVects, eigValues] = power_eig(A,2, direction='smallest first')
	print 'Eigenvectors : \n',  eigVects , '\n\n'
	print 'Eigenvalues : \n', eigValues , '\n----------\n'
	



	print 'Time difference '
	A = np.random.randn(400,400)
	A = A.dot(A.T)

	start_time = time.time() 
	[eigVects, eigValues] = power_eig(A,2, accuracy=0.00001)
	print("Power Method : %s seconds ---" % (time.time() - start_time))

	start_time = time.time() 
	[V,D] = eig_sorted(A)
	print("Eig Method : %s seconds ---" % (time.time() - start_time))

	#import pdb; pdb.set_trace()
	
	
	
	
