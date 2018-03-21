#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt



def gen_spiral(N,D,K, valid):
	X = np.zeros((N*K,D)) # data matrix (each row = single example)
	y = np.zeros(N*K, dtype='uint8') # class labels
	for j in xrange(K):
	  ix = range(N*j,N*(j+1))
	  r = np.linspace(0.3,2,N) # radius
	  t = np.linspace(j*4,(j+0.6)*4,N) + np.random.randn(N)*0.16 # theta
	  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
	  y[ix] = j
	
	
	labels = np.vstack((np.ones((N,1)), 2*np.ones((N,1)), 3*np.ones((N,1))))
	np.savetxt('spiral_arm' + valid + '.csv', X, delimiter=',', fmt='%d') 
	np.savetxt('spiral_arm_label' + valid + '.csv', labels, delimiter=',', fmt='%d') 
	
	plt.scatter(X[0:N, 0], X[0:N, 1], c='blue')
	plt.scatter(X[N:2*N, 0], X[N:2*N, 1], c='green')
	plt.scatter(X[2*N:3*N, 0], X[2*N:3*N, 1], c='red')
	plt.show()

def load_spiral():
	X = genfromtxt('spiral_arm.csv', delimiter=',')
	plt.scatter(X[0:100, 0], X[0:100, 1], c='blue')
	plt.scatter(X[100:200, 0], X[100:200, 1], c='green')
	plt.scatter(X[200:300, 0], X[200:300, 1], c='red')
	plt.show()

if __name__ == '__main__':
	N = 400 # number of points per class
	D = 2 # dimensionality
	K = 3 # number of classes

	gen_spiral(N,D,K, '_validation')
