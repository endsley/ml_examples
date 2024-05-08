#!/usr/bin/env python

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

#	This example code shows how you can 
#	use scipy to fit data to a beta distribution



# Generate 1000 samples from the beta distribution 
#	With a = 10, b = 2, 0 x shift, and scale of 2
X = 2*np.random.beta(10, 2, size=20000)

#	Fit and reproduce the parameters
[α,β,Δ,c] = beta.fit(X)
print('α: %.3f, β: %.3f, x shift: %.3f, data scaling: %.3f'%(α,β,Δ,c))

xvals = np.linspace(0,2,30)
yvals = beta.pdf(xvals, α, β, loc=Δ, scale=c)

#	plot the p(x) against the histogram
plt.title('α:%.3f, β:%.3f, Δ:%.3f, scale:%.3f'%(α,β,Δ,c))
plt.plot(xvals, yvals)
plt.hist(X, bins=100, density=True, alpha=0.5)
plt.show()

