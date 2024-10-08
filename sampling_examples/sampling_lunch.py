#!/usr/bin/env python

import numpy as np
from numpy import mean
from numpy import exp
from numpy import genfromtxt
from numpy import inf as ᨖ
from scipy.integrate import quad 

X = genfromtxt('lunch_wait_time.csv', delimiter=',')
x̄ = mean(X)
θ = 1/x̄


#### Part 1


def p(x):
	return θ*exp(-θ*x)
 
def f(x): return x*p(x)
  

μ, err = quad(f, 0, ᨖ ) 
print('Integration Vs Avg: %.3f, %.3f'%(μ,x̄))


#### Part 2

def ƒ(x): return (2*x + 1)*p(x)
x̄ = mean(2*X + 1)
  
μ, err = quad(ƒ, 0, ᨖ ) 
print('Integration Vs Avg: %.3f, %.3f'%(μ,x̄))





