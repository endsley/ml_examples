#!/usr/bin/env python

import numpy as np
from scipy.stats import norm
from numpy import genfromtxt
import matplotlib.pyplot as plt



#	Having μ, σ tells us $p(x) = \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{(x - \mu)^2}{2 \sigma^2}}$
X = genfromtxt('SAT.csv', delimiter=',')
[μ, σ] = norm.fit(X)

# Plot the histogram of the data
plt.hist(X, bins=40, density=True, color='skyblue', edgecolor='black');

# Plot the probability density function (PDF) of the normal distribution
x = np.linspace(600, 1600, 100)
y = norm.pdf(x, μ, σ)
plt.plot(x, y, color='red', linewidth=2)
plt.show()


