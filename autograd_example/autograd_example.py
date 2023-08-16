#!/usr/bin/env python
# Automatically find the gradient of a function
# Download the package at : https://github.com/HIPS/autograd

import autograd.numpy as np
from autograd import grad
#import random


#	Initial setup
n = 5
A = np.random.random((n,n))
W = np.random.random((n,1))




# Define a function Tr(WTA W), we know that gradient = (A+AT)W
def trance_quad(W, A): 
	return np.trace(np.dot(np.dot(np.transpose(W),A), W))

grad_foo = grad(trance_quad)       # Obtain its gradient function
print('Autogen Gradient : \n', grad_foo(W,A))
print('Theoretical Gradient : \n', np.dot((A+np.transpose(A)), W))


# ------------------------------
#n = 2
#A = (np.random.randint(-9,9,size=(n,n))).astype(float)
#W = (np.random.randint(-9,9,size=(n,1))).astype(float)

def mult_gaussian(W, A): 
	return np.exp(-np.trace(np.dot(np.dot(np.transpose(W),A), W)))

grad_foo = grad(mult_gaussian)       # Obtain its gradient function
print('Autogen Gradient : \n', grad_foo(W,A))
print('Theoretical Gradient : \n', -mult_gaussian(W,A)*(A+A.T).dot(W))



import pdb; pdb.set_trace()

