#!/usr/bin/env python

import numpy as np
from numpy import ones
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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


X = genfromtxt('stock_prediction_data_scaled.csv', delimiter=',')
poly = PolynomialFeatures(2)
Φ = poly.fit_transform(X)

n = Φ.shape[0]
d = Φ.shape[1]
η = 0.0001

y = genfromtxt('stock_price.csv', delimiter=',')
y = np.reshape(y, (n,1))
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
#	You can rewrite it into compact matrix vector form if you are good, note that $y$ is
#	a column vector of all labels. Here we used the advanced version. 
#	Look at the linear regression version for the standard derivative version.
#	The derivation can be found here:
#	https://github.khoury.northeastern.edu/chieh/course_page/blob/main/4420/lecture_4/compact_regression_derivative_derivation.pdf
#	$$f'(x) = \frac{2}{n} \Phi^{\top}(\Phi w - y)$$


mse_list = []
for i in range(300):
	fᑊ = Φ.T.dot(Φ.dot(w) - y)	# derivative in compact matrix form, so much easier.
	w = w - η*fᑊ				# gradient descent update w
	mse_list.append(f(w))		# record mse to plot later

plt.plot(mse_list)
plt.title('MSE over GD')
plt.xlabel('steps')
plt.ylabel('MSE')
plt.show()


# my stock price change prediction
ŷ = Φ.dot(w)
Y = np.hstack((ŷ, y))
print('Side by side comparison ŷ vs y') 
print(Y[0:20,:])
