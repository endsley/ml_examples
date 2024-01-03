#!/usr/bin/env python
# Automatically find the gradient of a function
# Download the package at : https://github.com/HIPS/autograd

import autograd.numpy as np
from autograd import grad

def f(x): 
	return np.log(2*x*x)/np.log(3) - 2*x*np.exp(3*x) + 2

def ᐁf(x):
	return 2/(x*np.log(3)) - 2*np.exp(3*x) - 6*x*np.exp(3*x)

auto_grad = grad(f)       # Obtain its gradient function

for i in range(10):
	x = np.random.randn()
	print('Auto ᐁf : %.3f, Theoretical ᐁf %.3f'%(auto_grad(x), ᐁf(x)))




