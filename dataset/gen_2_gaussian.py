#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

n = 200

x1 = np.random.randn(n, 2)
x2 = 0.6*np.random.randn(n, 2) + 5
X = np.vstack((x1,x2))



plt.plot(x1[:,0], x1[:,1], 'r.')
plt.plot(x2[:,0], x2[:,1], 'b.')
plt.show()
