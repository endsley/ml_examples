#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

X = np.array(	[	[1,1],
					[2,1],
					[1.5,0],
					[3,2]])

y = np.array([	[1],
				[1],
				[0],
				[2]])

Φ = np.array([	[1, 1,1,1],
				[8, 4,2,1],
				[3.375, 2.25, 1.5, 1],
				[27, 9,3,1]])
w = np.linalg.inv(Φ.T.dot(Φ)).dot(Φ.T).dot(y)
print(w)

x = np.reshape(np.arange(0,4, 0.1), (40,1))
Φ2 = np.hstack((x*x*x, x*x, x, np.ones((40,1))))
ŷ = Φ2.dot(w)


plt.scatter(X[:,0] ,y, color='red')
plt.plot(x,ŷ)
plt.ylim(-1,3)
plt.show()
