#!/usr/bin/env python

import numpy as np
from numpy import ones
from numpy import array
import matplotlib.pyplot as plt
from numpy import mean

xᑊ = array([[0],[1]])
xᒾ = array([[1],[1]])
X = np.vstack((xᑊ.T, xᒾ.T))
n = X.shape[0] # number of samples

A = array([[0,1],[1,1]])
y = array([[0],[2]])
w = array([[2],[2]]) # solution is 1,1
η = 0.1

#	The function we are trying to minimize is
#	$$f(x) = \frac{1}{n} \; \sum_i^n \; (w^{\top} \phi(x_i) - y_i)^2$$


def f(w):
	fₒ = 0						# function output
	for xᵢ, yᵢ in zip(X,y):
		xᵢ = np.reshape(xᵢ, (2,1))
		fₒ += (w.T.dot(xᵢ) - yᵢ)**2
	
	return ((1/n)*fₒ).item()
	

#	The equation for the gradient is 
#	$$f'(x) = \frac{2}{n} \; \sum_i^n \; (w^{\top} \phi(x_i) - y_i) \phi(x_i)$$


def fᑊ(w):
	ᐁf = np.zeros((2,1))
	for xᵢ, yᵢ in zip(X,y):
		xᵢ = np.reshape(xᵢ, (2,1))
		ᐁf += (w.T.dot(xᵢ) - yᵢ)*xᵢ
	return (2/n)*ᐁf

f_value_list = []
for i in range(20):
	w = w - η*fᑊ(w)				# gradient descent update w
	f_value_list.append(f(w))

print('Best w = \n', w)

# Display the plot
plt.plot(f_value_list)
plt.title('MSE over GD')
plt.xlabel('steps')
plt.ylabel('MSE')
plt.show()

