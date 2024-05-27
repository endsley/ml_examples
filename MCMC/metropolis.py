#!/usr/bin/env python

import numpy as np
from numpy import exp as e
from numpy.random import rand, exponential, normal
import matplotlib.pyplot as plt


#	Use metropolis to sample an exponential distribution
def p(x):
	try: 
		if x < 0: return 0
	except: x[x<0] = 0
	return e(-x)

def metropolis_sampler(n, μ, σ):
	samples = []
	
	while len(samples) != n:
		ᶍ = normal(μ, σ) # generate a new samples
		if rand() < p(ᶍ)/p(μ): 
			μ = ᶍ
		samples.append(μ)
		   
	return samples

# Parameters
n= 10000
μ = 0.5
σ = 0.5

# Generate samples
X = metropolis_sampler(n, μ, σ)
X2 = exponential(scale=1, size = n)


# Plot them out
plt.figure(figsize=(10,4))
plt.subplot(121)
xi = np.linspace(0.1,5,100)
plt.plot(xi,p(xi), color='red')
plt.hist(X, density=True, bins=30)
plt.title('Using Metropolis')

plt.subplot(122)
plt.plot(xi,p(xi), color='red')
plt.hist(X2, density=True, bins=30)
plt.title('Using Actual Exponential')

plt.show()
