#!/usr/bin/env python

import numpy as np


# random starting w 
w = np.array([[1],[1],[1]])
Φ = np.array([	[0,1,1],
				[1,0,1],
				[3,2,1],
				[3,3,1]])

y = np.array(	[[0],
				 [0],
				 [1],
				 [1]])

#	$$ q(x=1|w) = \frac{1}{1 + e^{-\phi(x)^{\top} w}}  $$
def q(x, w):
	return 1/(1 + np.exp(-x.T.dot(w)))

#	$$ \frac{dL}{dw} = \frac{1}{n} \sum_i (q(x_i=1|w) - p(x_i = 1)) \phi(x_i)$$
def dL(Φ, y, w):
	L = 0
	for i, j in enumerate(Φ):
		x = j.reshape(3,1)
		L += (q(x, w) - y[i])*x
	dL_dθ = L/len(Φ)	
	return dL_dθ

def gradient_descent(Φ, y, w, η):
	for i in range(2000):
		w = w - η*dL(Φ, y, w)
	return w


print('Starting Prediction (bad)')
for x in Φ: print(q(x, w))
w = gradient_descent(Φ, y, w, 0.1)

print('\nPrediction After GD (good)')
for x in Φ: print(q(x, w))

