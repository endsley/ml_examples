#!/usr/bin/env python

import numpy as np
from numpy import round


# random starting w 
w = np.array([[1],[1],[1]])
Φ = np.array([	[0,1,1],
				[1,0,1],
				[3,2,1],
				[3,3,1]])
n = 4
y = np.array(	[[0],
				 [0],
				 [1],
				 [1]])

#	$$ q(x=1|w) = \frac{1}{1 + e^{-\phi(x)^{\top} w}}  $$
def q(x=None, X=None, w=None):
	if X is None:
		return 1/(1 + np.exp(-x.T.dot(w)))
	else:
		return 1/(1 + np.exp(-X.dot(w)))

#	Using summation derivative
#	$$ \frac{dL}{dw} = \frac{1}{n} \sum_i (q(x_i=1|w) - p(x_i = 1)) \phi(x_i)$$
def dL_1(Φ, y, w):
	L = 0
	for i, j in enumerate(Φ):
		x = j.reshape(3,1)
		L += (q(x, None, w) - y[i])*x
	dL_dθ = L/n
	return dL_dθ


#	Using Compact Matrix derivative
#	$$ \frac{dL}{dw} = \frac{1}{n} \Phi^{\top} (σ(X) - y) $$
def dL_2(Φ, y, w):
	return Φ.T.dot(q(None, Φ, w) - y)/n


def gradient_descent(Φ, y, w, η):
	for i in range(2000):
		#w = w - η*dL_1(Φ, y, w)
		w = w - η*dL_2(Φ, y, w)
	return w


print('Starting Prediction (bad)')
for x in Φ: print(q(x, None, w))
w = gradient_descent(Φ, y, w, 0.1)

print('\nPrediction After GD (good)')
for x in Φ: print(q(x, None, w))

print('\nFinal Prediction is displayed with rounding')
for x in Φ: print(round(q(x, None, w)))

