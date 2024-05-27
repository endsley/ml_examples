#!/usr/bin/env python

import numpy as np
from numpy import exp as e
from numpy.random import rand, exponential
import matplotlib.pyplot as plt


#	Use metropolis to sample an exponential distribution
def p(x):
	try: 
		if x < 0: return 0
	except: x[x<0] = 0
	return e(-x)

def metropolis_sampler(n, μ, σ, start_point=30):
	samples = []
	
	while len(samples) != n + start_point:
		ᶍ = np.random.normal(μ, σ) # generate a new samples
		if rand() < p(ᶍ)/p(μ): 
			samples.append(ᶍ)
			μ = ᶍ
		   
	return samples[start_point:]

# Parameters
n= 20000
μ = 0.0
σ = 1

# Generate samples
X = metropolis_sampler(n, μ, σ)
X2 = exponential(scale=1, size = 20000)


# Plot them out
plt.figure(figsize=(10,4))
plt.subplot(121)
xi = np.linspace(0.1,5,100)
yi = p(xi)
plt.plot(xi,yi, color='red')
plt.hist(X, density=True, bins=30)
plt.title('Using Metropolis')

plt.subplot(122)
plt.plot(xi,yi, color='red')
plt.hist(X2, density=True, bins=30)
plt.title('Using Actual Exponential')

plt.show()
