#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize

def ƒ(x): # The objective 
	xₐ = x[0]
	xᵦ = x[1]
	return 0.5*xₐ**2 + xᵦ**2

def ℊₗ(x): # The equality constraint
	xₐ = x[0]
	xᵦ = x[1]
	return xₐ - 3*xᵦ

def ℊշ(x): # The inequality constraint
	xᵦ = x[1]
	return xᵦ - 1
	
constraints = [{'type':'ineq', 'fun':ℊₗ}, {'type':'ineq', 'fun':ℊշ}]

xₒ = [4,4]
result = minimize(ƒ, xₒ , constraints=constraints)
print('Starting at (4,4), the solution gets stuck at [4,1]')
print(result['x'])

