#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize


#	Equality constraint
def obj(x):
	return x**2

def eq_constraint(x):
	return x**2 - x - 2

constraints = [{'type':'eq', 'fun':eq_constraint}]

#This problem has 2 possible solutions 
#	depending on where x0 starts, you'll end up with a different solution.
result = minimize(obj, -2, constraints=constraints)
print(result)

result = minimize(obj, 1, constraints=constraints)
print(result)



#	Multivariate Equality constraint
def obj(x):	# The input x vector gets flattened
	x = np.reshape(x,(2,1))
	c = np.array([[2],[1]])
	y = (c*x).T.dot(x)
	return y

def eq_constraint(x):
	# The input x vector gets flattened
	x = np.reshape(x,(2,1))
	O = np.ones((2,1))
	y = 1 - O.T.dot(x).item()
	return y

x0 = np.array([[3],[2]])
constraints = [{'type':'eq', 'fun':eq_constraint}]
result = minimize(obj, x0, constraints=constraints)
print(result)


