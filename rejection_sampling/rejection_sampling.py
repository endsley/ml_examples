#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, expon
from numpy.random import rand


#	The rejection sampling idea
#	- Our goal is to generate samples from $p(x)$ 
#	- However, we can do it directly, so we use a proposal distribution $q(x)$
#	- By generating samples from $q(x)$, we can use rejection sampling to get samples from $p(x)$.

#	The rejection sampling Algorithm
#	- Step 1: Pick a constant k to multiply such that $k q(x) > p(x)$
#	- Step 2: Sample from our proposal distribution, let's call the sample $x_1$ 
#	- Step 3: If $kq(x_1)*rand() < p(x_1)$ then we accept the sample $x_1$ as a newly generated sample
#	- Step 4: Repeat this to continue to gather samples. 



#	 proposal distribution
def kq(x):
	return 2*expon.pdf(x, loc=0, scale=6)


def reject_sampling(n):
	X = []
	while len(X) < n:
		s = expon.rvs(loc=0, scale=6)
		if kq(s)*rand() < chi2.pdf(s,3):
			X.append(s)
	return X

#	Notice that kq(x) is always larger than $p(x)$
x = np.linspace(0, 20, 100)
y = chi2.pdf(x,3)
y2 = kq(x)

X = reject_sampling(10000)

#	Notice that the histogram generated fix the chi2 distribution perfectly
plt.plot(x,y)
plt.hist(X, density=True, bins=40, alpha=0.7)
plt.plot(x,y2)
plt.title('Rejection Sampling Example')
plt.show()

