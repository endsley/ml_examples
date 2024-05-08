#!/usr/bin/env python

import numpy as np
from numpy import exp as e
from numpy import inf as ᨖ
from scipy.integrate import quad 
from scipy.special import gamma as Γᵦ


#	The $\Gamma(z)$ function is defined as 
#	$$\int_0^{\infty} \; t^{z-1} \; e^{-t} \; dt$$
#	It turns out that if $z$ happens to an integer, the integral simplies into
#	$$\Gamma(z) = (z - 1)!$$


#	My own defined Γ function.
def Γₐ(z):
	def γ(t): return t**(z-1)*e(-t)
	area , err = quad(γ, 0, ᨖ ) 
	return area



print('My own Γ implementation result : %.4f' % Γₐ(4.3))
print('Numpy Γ implementation result : %.4f\n' % Γᵦ(4.3))

print('My own Γ implementation result : %.4f' % Γₐ(4))
print('Numpy Γ implementation result : %.4f\n' % Γᵦ(4))
print('Integer z result : %.4f\n' % (3*2))

