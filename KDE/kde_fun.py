#!/usr/bin/env python

import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

X = np.array([1,1,2,3])
Զπ = 2*np.pi
σ = 0.4
n = len(X)


def p(x): # This is the kde p(x) distribution
	def gaussian(μ, σ, x): return np.exp(-0.5 * ((x - μ) / σ)**2) / (σ * sqrt(Զπ))
	px = 0
	for ӽ in X: 
		px += gaussian(ӽ, σ, x)
	return px/n

# Generate x-axis values for the PDF plot
x = np.linspace(0, 5, 100)

# Compute the PDF values for the mixture of two Gaussian distributions
y = p(x)

# Plot the PDF of the mixture distribution
plt.plot(x, y, 'r-', linewidth=2)

# Set plot title and labels
plt.title('p(x) with KDE')
plt.xlabel('x')
plt.ylabel('Probability Density')

# Display the plot
plt.show()

