#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp as e

θ = 1/3
n = 50000
X = np.random.exponential(scale=3, size=n)

#	The corresponding equation given $\theta = 1/3$
#	$$ p(x) = \theta e^{-\theta x} $$

x = np.linspace(0, 15, 100)
y = θ*e(-θ*x)


print('μ = %.3f'%np.mean(X))
plt.hist(X, 30, density=True, alpha=0.7)
plt.plot(x,y)
plt.show()

#	Now we generate from uniform distribution
X = np.random.rand(n)
print('P(x < 0.3) = %.3f'% (len(X[X<0.3])/n))
print('P(x < 0.6) = %.3f'% (len(X[X<0.6])/n))
