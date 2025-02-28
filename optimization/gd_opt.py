#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
from numpy import max
import sys
np.set_printoptions(precision=2)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)



#	In this example, we are trying solve a constrained optimization problem using GD
#	We changed the original objective for the equality constraint
#	$$\min_x \; x^2 \quad s.t: x^2 - x - 2 = 0, \; x \le 0 $$


eq = ['h(x) ⵐ 0 : failed', 'h(x) = 0 : close enough']
ineq = ['g(x) > 0 : failed', 'g(x) < 0 : close enough']
𝕀 = lambda x: int((x > 0))

η = 0.000000001
def f(x):
	return x**2

def h(x):	# equality constraint
	return x**2 - x - 2

def L(x, λ, γ):
	return x**2 + λ*h(x)**2 + γ*max(0,x)

def ᐁL(x, λ, γ):
	return 2*x + 2*λ*h(x)*(2*x - 1) + γ*𝕀(x)


#	Notice how while the equality constraint did okay, the Inequality didn't
λ = 500 
γ = 200
x = 3
for i in range(1000000): 
	x = x - η*ᐁL(x, λ, γ) # GD

print('Given λ=%.2f, γ=%.2f, %s , %s'%(λ, γ, eq[int(norm(h(x)) < 0.03)], ineq[int(x < 0)]))
print('f(x): %.4f, \t h(x): %.4f, \t x: %.4f, \t ᐁL: %.8f\n'%(f(x), h(x), x, ᐁL(x, λ, γ)))



