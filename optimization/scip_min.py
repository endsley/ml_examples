#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize

#	Another example of Equality constraint
def ƒ(x): # The objective 
	xₐ = x[0]
	xᵦ = x[1]
	return xₐ + xᵦ

def ɦ(x): # The equality constraint
	xₐ = x[0]
	xᵦ = x[1]
	return xₐ**2 + xᵦ**2 - 1

constraints = [{'type':'eq', 'fun':ɦ}]



#	Notice that depending on your starting point
#	python might only give you the local optimum


xₒ = [-4,-4]
result = minimize(ƒ, xₒ, constraints=constraints)
print(result)




