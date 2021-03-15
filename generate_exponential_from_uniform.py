#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def exponential_from_uniform(x, λ):
	return -(1/λ)*np.log(1-x)


x = np.random.rand(30000)
expD = exponential_from_uniform(x, 1)

n, bins, patches = plt.hist(expD, 30, facecolor='blue', alpha=0.5)
plt.show()

