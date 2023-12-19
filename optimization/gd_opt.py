#!/usr/bin/env python

import numpy as np
import sys
np.set_printoptions(precision=2)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)
#np.set_printoptions(threshold=sys.maxsize)


λ = 2
γ = 10
η = 0.001
x = -0.5
L_before = 100

def f(x):
	return x**2

def c1(x):	# equality constraint
	return x**2 - x + 2

def L(x):
	return x**2 + λ*c1(x)**2 + γ*x

def ᐁL(x):
	return 2*x + 2*λ*c1(x)*(2*x - 1) + γ

for i in range(2000):
	x = x - η*ᐁL(x)
	print('Obj: %.4f, \t c1: %.4f, \t x: %.4f, \t ᐁL: %.8f'%(L(x), c1(x), x, ᐁL(x)))
	#if L(x) < (L_before - 0.00001):
	#	print('Obj: %.4f, \t c1: %.4f, \t x: %.4f, \t ᐁL: %.8f'%(L(x), c1(x), x, ᐁL(x)))
	#	L_before = L(x)
