#!/usr/bin/env python
# Automatically find the gradient of a function
# Download the package at : https://github.com/HIPS/autograd

import autograd.numpy as np
from autograd.numpy import exp as e
from autograd.numpy import power
from autograd.numpy import transpose as T
from autograd import grad
#import random


#	Initial setup
x = np.array([[1],[2]])
W = np.array([	[0.8,0.3],
				[0.1,0.3],
				[0.3,0.3]])

def f(w): 
	v = e(-np.dot(T(w),x))
	return 1/(1+v)
	
def ߜf(w): 
	v = e(-np.dot(T(w),x))
	return (v/((1+v)*(1+v)))*x


pf = grad(f)       # Obtain its gradient function
for w in W:
	w = np.reshape(w, (2,1))
	print(w)
	print('Autogen Gradient : ', pf(w).flatten())
	print('Theoretical : ', ߜf(w).flatten()) 
	print('\n\n')

