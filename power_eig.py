#!/usr/bin/python

import numpy as np
import time 

def multiply_until_converge(X, f):
	for p in range(100):
		new_f = X.dot(f)
		new_f = new_f/np.max(new_f)
		if np.linalg.norm(new_f - f) < 0.0001*np.linalg.norm(f):
			break
		else:
			f = new_f

	print 'p : ' , p
	f = f/np.linalg.norm(f)
	return f

def power_eig(A, accuracy=0.0001):
	if A.shape[0] != A.shape[1]:
		print 'Error : A must be a squared matrix'
		return

	N = np.linalg.norm(A)
	d = A.shape[0]
	X = N*np.eye(d) - A
	
	#[V,D] = np.linalg.eig(X)
	#print V , '\n\n'
	#print D
	
	f = np.random.randn(d,1)
	f = multiply_until_converge(X, f)
	eig_value = N - f.T.dot(X).dot(f)
	print f , '\n'
	print eig_value
	print 'dddddddddddddd'


	X = X - eig_value*np.eye(d)
	f2 = np.random.randn(d,1)
	f2 = multiply_until_converge(X, f2)
	eig_value = N - f2.T.dot(X).dot(f2)

	print f , '\n'
	print eig_value
	print 'dddddddddddddd'


np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)


#A = np.array([[1,2,4],[4,0,6],[1,1,1]]);
A = np.random.randn(4,4)
A = A.dot(A.T)
start_time = time.time() 
power_eig(A)


	
start_time = time.time() 
[D,V] = np.linalg.eig(A)
print V , '\n\n'
print D , '\n----------\n'


#import pdb; pdb.set_trace()




