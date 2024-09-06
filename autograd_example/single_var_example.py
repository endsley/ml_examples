#!/usr/bin/env python
# Automatically find the gradient of a function
# Download the package at : https://github.com/HIPS/autograd

import autograd.numpy as np
from autograd.numpy import log as ln
from autograd.numpy import exp
from autograd import grad


#	Given the function
#	$$f(x) = \log_3(2x^2) - 2x e^{3x} + 2$$
#	The derivative should be
#	$$f'(x) = \frac{2}{x \ln{3}} - 2 e^{3x} - 6 x e^{3x} $$


def f(x): 
	return ln(2*x*x)/ln(3) - 2*x*exp(3*x) + 2

def ᐁf(x):
	return 2/(x*ln(3)) - 2*exp(3*x) - 6*x*exp(3*x)

auto_grad = grad(f)  # Automatically obtain the gradient function

for i in range(10):
	x = np.random.randn()
	print('Auto ᐁf : %.3f, Theoretical ᐁf %.3f'%(auto_grad(x), ᐁf(x)))




