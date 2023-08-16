#!/usr/bin/env python
# Automatically find the gradient of a function
# Download the package at : https://github.com/HIPS/autograd

import autograd.numpy as np
from autograd import grad


# Define a function Tr(WTA W), we know that gradient = (A+AT)W
def relu(w, x): 
	v = np.dot(np.transpose(w),x)
	return np.maximum(0, v)

def grad_relu(w,x):
	if w.T.dot(x) > 0:
		return x
	else:
		return 0

for i in range(3):
	#	Initial setup
	x = np.random.randn(2,1)
	w = np.random.randn(2,1)
	
	grad_foo = grad(relu)       # Obtain its gradient function
	print('w.dot(x) = %.3f'%x.T.dot(w))
	print('Autogen Gradient : \n', grad_foo(w,x), '\n')
	print('Theoretical Gradient : \n', grad_relu(w,x), '\n\n\n')


