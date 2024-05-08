#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
import sys
np.set_printoptions(precision=2)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)



#	In this example, we are trying solve a constrained optimization problem using GD
#	We changed the original objective for the equality constraint
#	$$\min_x \; x^2 \quad s.t: x^2 - x - 2 = 0, \; x \le 0 $$
#	This example demonstrates that even if you adjust the Lagrangian to use GD
#	It could take you a while to finding γ and λ. 
#	Depending on the problem, you could be looking at millions of possibilities. 



eq = ['h(x) ⵐ 0 : failed', 'h(x) = 0 : close enough']
ineq = ['g(x) > 0 : failed', 'g(x) < 0 : close enough']

η = 0.001
def f(x):
	return x**2

def h(x):	# equality constraint
	return x**2 - x - 2

def L(x, λ, γ):
	return x**2 + λ*h(x)**2 + γ*x

def ᐁL(x, λ, γ):
	return 2*x + 2*λ*h(x)*(2*x - 1) + γ



#	Here we will print out if the GD solution satisfies the constraints



for λ in range(4):
	for γ in range(4):
		x = 3
		for i in range(3000): 
			x = x - η*ᐁL(x, λ, γ) # GD
#
		print('Given λ=%.2f, γ=%.2f, %s , %s'%(λ, γ, eq[int(norm(h(x)) < 0.03)], ineq[int(x < 0)]))
		print('f(x): %.4f, \t h(x): %.4f, \t x: %.4f, \t ᐁL: %.8f\n'%(f(x), h(x), x, ᐁL(x, λ, γ)))


