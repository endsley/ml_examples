#!/usr/bin/python

import numpy as np
from scipy.optimize import minimize

constant = [1,2,-1]


def Lagrange_W(W, X, L1, L2, Z, W_shape):	
	W2 = W.reshape(W_shape)
	one_matrix = np.ones((W_shape[1],W_shape[1]))
	eye_matrix = np.eye(W_shape[1])

	cost_foo = 0
	for m in range(len(X)):
		x = X[m]
		cost_foo = cost_foo - constant[m]*np.exp(-x.T.dot(W2).dot(W2.T).dot(x))

	#cost_foo = -np.exp(-x1.T.dot(W2).dot(W2.T).dot(x1))-2*np.exp(-x2.T.dot(W2).dot(W2.T).dot(x2))

	Lagrange = np.trace(L1.dot(W2.T.dot(Z) - eye_matrix)) + np.sum(L2.T*(W2 - Z))

	term1 = ( Z.T.dot(W2) - eye_matrix )
	term2 =  W2 - Z 
	

	Aug_lag = np.trace(one_matrix.dot(term1*term1)) + np.sum(term2*term2)
	foo = cost_foo + Lagrange + Aug_lag

	return foo

def Lagrange_Z(Z, L1, L2, W, Z_shape):
	
	Z2 = Z.reshape(Z_shape)
	one_matrix = np.ones((Z_shape[1],Z_shape[1]))
	eye_matrix = np.eye(Z_shape[1])

	#import pdb; pdb.set_trace()
	Lagrange = np.trace(L1.dot(W.T.dot(Z2) - eye_matrix)) + np.sum(L2.T*(W - Z2))

	term1 = ( Z2.T.dot(W) - eye_matrix )
	term2 =  W - Z2 
	

	Aug_lag = np.trace(one_matrix.dot(term1*term1)) + np.sum(term2*term2)
	foo = Lagrange + Aug_lag

	return foo



X = []
X.append(np.array([1,0,1]))
X.append(np.array([3,2,1]))
X.append(np.array([3,4,-1]))

#x1 = np.array([1,0,1])
#x2 = np.array([3,2,1])

Z = np.array([[1,0],[0,1],[0,0]])
W = np.array([[1,0], [0,1],[0,0]])
L1 = np.array([[1,0], [0,2]])
L2 = np.array([[2,3,1],[0,0,1]])
L3 = np.array([[2,0,1],[0,0,1]])
L = np.append(L1, L2, axis=1)

zi = Z.shape[0]
zj = Z.shape[1]
stay_in_loop = True
loop_count = 0



while stay_in_loop:
	result_w = minimize(Lagrange_W, W, method='nelder-mead', args=(X,L1,L2,Z, W.shape), options={'xtol': 1e-6, 'disp': True})
	W = result_w.x.reshape(W.shape)
	
	result_z = minimize(Lagrange_Z, Z, method='nelder-mead', args=(L1,L2,W, Z.shape), options={'xtol': 1e-6, 'disp': True})
	Z = result_z.x.reshape(Z.shape)
	
	#print Z
	A = np.append(Z.T, np.eye( zi ), axis=0)
	B = np.append(np.zeros(Z.T.shape), np.eye( zi ), axis=0)
	C = np.append(np.eye(zj), np.zeros(Z.shape), axis=0)
	L = L + (A.dot(W)-B.dot(Z)-C).T
	
	L1 = L[:,0:zj]
	L2 = L[:,zj:]	
	
	loop_count += 1
	if np.abs(np.sum(W.T.dot(Z) - np.eye(zj))) < 0.001: 
		print('Exit base on threshold')
		stay_in_loop = False
	if loop_count > 200: 
		print('Exit base on loop_count')
		stay_in_loop = False


print result_w
import pdb; pdb.set_trace()
