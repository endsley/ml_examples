#!/usr/bin/env python

import numpy as np
from scipy.special import gamma as Γ
from numpy import exp as e
from numpy import inf as ᨖ
from numpy.random import rand, normal
from scipy.integrate import quad 
from scipy.stats import invgamma
import matplotlib.pyplot as plt



def p(x, a=3, b=1):	#inverse gamma distribution
	try: 
		if x < 0: return 0
	except: x[x<0] = 0
	return (b**a/Γ(a))*(x)**(-a-1)*e(-b/x)






#	Confirm that the area is == 1
a = 3
b = 1
area , err = quad(p, 0, ᨖ , args=(a, b)) 
print(area)





#	Use metropolis to generate samples 
def metropolis_sampler(n, μ=0.5, σ=0.5):
	samples = []
	
	while len(samples) != n:
		ᶍ = normal(μ, σ) # generate a new samples
		if rand() < p(ᶍ)/p(μ): μ = ᶍ
		samples.append(μ)	# you either accept ᶍ or accept μ 
		   
	return samples


n= 10000
X = metropolis_sampler(n)
Ẋ = invgamma.rvs(a=a, scale=b, size=n)

# Plot them out
plt.figure(figsize=(10,4))

plt.subplot(121)
xi = np.linspace(0.1,3,100)
plt.plot(xi, p(xi), color='red')
plt.hist(X, density=True, bins=100)
plt.title('Using Metropolis')
plt.xlim(0, 3)

plt.subplot(122)
plt.plot(xi,p(xi), color='red')
plt.hist(Ẋ, density=True, bins=130)
plt.title('Using Python Gamma sampler')
plt.xlim(0, 3)


plt.show()
