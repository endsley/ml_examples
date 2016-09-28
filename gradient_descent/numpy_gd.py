#!/usr/bin/python

#	Numpy approach to solve f(x,y) = (x-3)^2 + (y-1)^2

import numpy as np
from scipy.optimize import minimize

def cost_fun(x, arg1, arg2):
	return arg1*pow((x[0] - 3), 2) + arg2*pow((x[1] - 1), 2) 

x0 = np.array([0,0])
res = minimize(cost_fun, x0, method='nelder-mead', args=(3,4), options={'xtol': 1e-8, 'disp': True})
print res
