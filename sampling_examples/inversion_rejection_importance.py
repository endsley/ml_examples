#!/usr/bin/env python

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad 
from numpy import sum, mean, linspace
from numpy import exp as e
from numpy.random import exponential
from numpy.random import rand
import matplotlib.pyplot as plt 
 


####	Directly solving the integral
#	$$\int_0^1 \; (x+1) x^2 \; dx$$
def f(x): return (x+1)*x**2
I, err = quad(f, 0, 1) 
print('Integral using Numpy: %.3f'%I)


####	Using inversion sampling
#	Start by using the definition of CDF function to find the inverse CDF function $F^{-1}$
#	$$F(α) = \int_0^{\alpha} x^2 \; dx = \frac{1}{3} \alpha^3 \quad \implies \quad x = (3v)^{1/3}.$$
#	This implies that if you generate uniform samples from 0 to 1, we can plug into v to generate sample for p(x)
#	This integral becomes
#	$$\int_0^1 \; (x+1) p(x \; \text{and}\; 0 \le x \le 1) \; dx = \int_0^1 \; (x+1) p(x | 0 \le x \le 1)  \underbrace{p(0 \le x \le 1)}_{\rho} \; dx \approx \frac{\rho}{n} \sum_i \; (x+1)$$
n = 20000
X = rand(n)
X2 = np.power(3*X, 1/3)
ρ = sum(X2 < 1)/len(X)
X2 = X2[X2 < 1]
print('Integral using Inversion Sampling : %.3f'% (mean(X2 + 1)*ρ))

 

####	Using Rejection sampling
#	We are using an exponential distribution to approximate $p(x) = x^2$ 
#	First we need to figure out the k value by plotting it out.
#	notice that if k=10, then we have kq(x) > p(x) to alway have a rejection region between 0 and 1.
#	We can pick any $\theta$ for exponential distribution, to make it easier let's just pick $\theta=1$
def p(x): 
	try:
		x[x<0] = 0
		x[x>1.44225] = 0
	except:
		if x < 0: return 0
		if x > 1.44225: return 0
	return x**2

k = 10
x = linspace(0,1.44225, 20)
y1 = p(x)
y2 = k*e(-x)

plt.title('Plot shows that $kq(x) > p(x)$')
plt.plot(x,y1, color='blue')
plt.plot(x,y2, color='red')
plt.show()

#	Now we can generate exponential distribution
X2 = []

while len(X2) < n:
	u = exponential(scale=1, size=1).item()
	kq = k*e(-u)
	pu = p(u)
	v = kq*rand()
	if v < pu: X2.append(u)	# accept 

X2 = np.array(X2)
plt.hist(X2,bins = 40, alpha=0.7,color='blue',density=True)
plt.title('Histogram of $x^2$ distribution')
plt.show()

ρ = sum(X2 < 1)/len(X2)	# same logic applies here for ρ
X2 = X2[X2 < 1]
print('Integral using Rejection Sampling : %.3f'% (mean(X2 + 1)*ρ))


####	Using Importance sampling
#	$$ \int_0^1 \; (x+1) p(x) \; dx = \int_0^1 \; (x+1) \frac{p(x)}{q(x)} q(x) \; dx $$
#	In this case, we have a simple $q(x) = 1$ so
#	$$	\int_0^1 \; (x+1) \frac{x^2}{1} q(x) \; dx \approx \frac{1}{n} \sum_i \; (x+1) x^2$$
X = rand(n)
print('Integral using importance sampling: %.3f'% (mean((X+1)*X**2)))
