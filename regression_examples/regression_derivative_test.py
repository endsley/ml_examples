#!/usr/bin/env python

import autograd.numpy as np
from autograd import grad
#import random


#	Initial setup
x = np.array([[2],[1]])
w = np.array([[1],[1]])
y = 1


# since n can be any value, I just set it to 1
# do a more advance once if you are ambitious
def f(w): 
	return (0.5)*(np.dot(w.T,x) - y)**2


def df(w): 
	return (w.T.dot(x) - y)*x

grad_foo = grad(f)       # Obtain its gradient function
print('Autogen Gradient : \n', grad_foo(w))
print('Theoretical Gradient : \n', df(w))
