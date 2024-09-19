#!/usr/bin/env python

import numpy as np
from numpy import ones
from numpy import array
import matplotlib.pyplot as plt
from numpy import mean

xᑊ = array([[0],[1]])
xᒾ = array([[1],[1]])
yᑊ = 0
yᒾ = 2

n = 2 # number of samples
η = 0.2 	# this is the step size

# You can initialize w at any point, solution is at 2,0
w = array([[2],[2]]) 


#	The function we are trying to minimize is
#	$$f(x) = \frac{1}{n} \; \sum_i^n \; (w^{\top} x_i - y_i)^2$$


def f(w):
	return (1/n)*((w.T.dot(xᑊ) - yᑊ)**2 + (w.T.dot(xᒾ) - yᒾ)**2)
	

#	The equation for the gradient is 
#	$$f'(x) = \frac{2}{n} \; \sum_i^n \; (w^{\top} x_i - y_i) x_i$$


def fᑊ(w):
	return (2/n)*((w.T.dot(xᑊ) - yᑊ)*xᑊ + (w.T.dot(xᒾ) - yᒾ)*xᒾ)

f_value_list = []
for i in range(100):
	w = w - η*fᑊ(w)				# gradient descent update w
	f_value_list.append(f(w).item())

print('Best w = \n', w)

# Display the plot
plt.plot(f_value_list)
plt.title('MSE over GD')
plt.xlabel('steps')
plt.ylabel('MSE')
plt.show()

