#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# Define the exponential distribution function
def exponential(x, beta):
	try: 
		if x < 0: return 0
	except: x[x<0] = 0

	return beta * np.exp(-beta * x)

# Define the proposal distribution (in this case, a Gaussian)
def proposal(x, sigma):
	return np.random.normal(x, sigma)

# Metropolis-Hastings algorithm
def metropolis_hastings(iterations, beta, sigma, x0):
	samples = [x0]
	x_current = x0
	for i in range(iterations):
		x_proposed = proposal(x_current, sigma)
		acceptance_ratio = exponential(x_proposed, beta) / exponential(x_current, beta)
		if np.random.uniform(0, 1) < acceptance_ratio:
			x_current = x_proposed
		samples.append(x_current)
	return samples

# Parameters
iterations = 10000
beta = 1.0  # Rate parameter of the exponential distribution
sigma = 0.5  # Standard deviation of the proposal distribution
x0 = 0.5  # Initial value

# Generate samples using Metropolis-Hastings algorithm
samples = metropolis_hastings(iterations, beta, sigma, x0)

# Plot histogram of samples
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')

# Plot the true exponential distribution for comparison
x_values = np.linspace(0, 5, 100)
plt.plot(x_values, exponential(x_values, beta), 'r-', label='True Distribution')

plt.title('Metropolis-Hastings Sampling of Exponential Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

