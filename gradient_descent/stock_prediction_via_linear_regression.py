#!/usr/bin/env python

import numpy as np
from numpy import ones
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from numpy import genfromtxt
from numpy import mean
from numpy.random import randn
import sys

#	these make printing nicer
np.set_printoptions(precision=4)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)



X = genfromtxt('stock_prediction_data.csv', delimiter=',')
X = preprocessing.scale(X)
n = X.shape[0]
η = 0.01

y = genfromtxt('stock_price.csv', delimiter=',')
y = np.reshape(y, (n,1))

Ⅱ = ones((n,1))	# this is a column vector of 1s
Φ = np.hstack((X,Ⅱ))
d = Φ.shape[1]
w = randn(d,1)


#	The function we are trying to minimize is
#	$$\min_w \; f(x) = \frac{1}{n} \; \sum_i^n \; (w^{\top} \phi(x_i) - y_i)^2$$


def f(w):
	fₒ = 0						# function output
	for ϕᵢ, yᵢ in zip(Φ,y):
		ϕᵢ = np.reshape(ϕᵢ, (d,1))		# make sure the ϕᵢ is in column format
		fₒ += (w.T.dot(ϕᵢ) - yᵢ)**2
	return ((1/n)*fₒ).item()	# this is the mse
	

#	The equation for the gradient is 
#	$$f'(x) = \frac{2}{n} \; \sum_i^n \; (w^{\top} \phi(x_i) - y_i) \phi(x_i)$$
#	You can rewrite it into compact matrix vector form if you are good, note that $y$ is a column vector of all labels.
#	$$f'(x) = \frac{2}{n} \Phi^{\top}(\Phi w - y)$$
#	[The derivation can be found here](https://github.khoury.northeastern.edu/chieh/course_page/blob/main/4420/lecture_4/compact_regression_derivative_derivation.pdf)	


def ᐁf(w):
	advanced_approach = False
	if advanced_approach:
		return (2/n)*Φ.T.dot(Φ.dot(w) - y) # derivative in compact matrix form
	else:
		grads = np.zeros((d, 1))	
		for ϕᵢ,yᵢ in zip(Φ,y):	# loop through both x and y each sample
			ϕᵢ = np.reshape(ϕᵢ, (d,1)) # make sure it is in column format
			grads += (w.T.dot(ϕᵢ) - yᵢ)*ϕᵢ
	return (2/n)*grads

mse_list = []
for i in range(200):
	w = w - η*ᐁf(w)				# gradient descent update w
	mse_list.append(f(w))

plt.plot(mse_list)
plt.title('MSE over GD')
plt.xlabel('steps')
plt.ylabel('MSE')
plt.show() 						# Display the plot


# my stock price change prediction
ŷ = Φ.dot(w)
Y = np.hstack((ŷ, y))
print('Side by side comparison ŷ vs y') 
print(Y[0:20,:])
