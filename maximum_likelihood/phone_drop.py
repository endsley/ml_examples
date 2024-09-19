#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon
from numpy import exp as ē
from numpy import sum as Σ
from numpy import array
from numpy import log as ln
from numpy import genfromtxt
from numpy import mean


X = genfromtxt('time_until_phone_drop.csv', delimiter=',')
n = X.shape[0]
μ = mean(X)

def ln_p(θ): # the log likelihood of joint
	return n*ln(θ) - θ*Σ(X)

def ᐁln_p(θ): # derivative of the log likelihood 
	return (n/θ - Σ(X))

θ = 2
η = 0.0001

for i in range(100):
	θ = θ + η*ᐁln_p(θ)

print('Using the mean method, the best θ = 1/μ = %.3f'%(1/μ))
print('Using Maximum Likelihood, the best θ = %.3f'%θ)
print('Probability of drop phone within 2 years %.3f'%expon.cdf(2, 0, μ))
print('Time until 90 percent of the population dropped their phone %.3f'%expon.ppf(0.9, 0, μ))


#	draw the histogram and the line
plt.hist(X, bins=50, density=True, color='skyblue', edgecolor='black')
x = np.linspace(0,15, 100)
y = θ*ē(-θ*x)
plt.plot(x, y, color='red', linewidth=2)
plt.show()


