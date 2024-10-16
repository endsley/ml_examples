#!/usr/bin/env python

from scipy.integrate import quad 
from numpy import exp as ē
from numpy import log as ln
from numpy import genfromtxt
from numpy import mean


X = genfromtxt('time_until_phone_drop.csv', delimiter=',')
n = X.shape[0]
μ = mean(X)
θ = 1/μ

#	Since this is an exponential distribution, we know that $p(x)$ and $H$ are
#	$$ p(x) = \theta e^{-\theta x} \quad \text{where} \quad \theta = 1/\mu$$
#	$$ H = - \int_0^{\infty} p(x) \log \left( p(x) \right) \approx \frac{1}{n} \sum_i \; -ln(p(x)) \quad \text{where} \quad - ln(p(x)) = ln\left( \frac{1}{p(x)} \right)$$.

def p(x): return θ*ē(-θ*x)
def f(x): return -p(x)*ln(p(x))
def g(x): return -ln(p(x))  

Ĥ, err = quad(f, 0, 50) 
print('Entropy by numpy : %.3f'%Ĥ) 
print('Entropy by averaging samples : %.3f'%mean(g(X))) 


