#!/usr/bin/env python

import numpy as np
from numpy import ones
from numpy import array
import matplotlib.pyplot as plt
from numpy import mean

A = array([[0,1],[1,1]])
y = array([[0],[2]])
w = array([[2],[2]]) 
η = 0.1

def f(w):
	p = (np.dot(A,w) - y)
	return np.dot(np.transpose(p), p).item()
	

#	The equation for the gradient is 
#	$$\frac{df}{dw} = 2A^{\top}(Aw - y)$$


f_value_list = []
for i in range(100):
	fᑊ = 2*A.T.dot(A.dot(w) - y) 
	w = w - η*fᑊ				# gradient descent update w
	f_value_list.append(f(w))

print('Best w = \n', w)

# Display the plot
plt.plot(f_value_list)
plt.title('MSE over GD')
plt.xlabel('steps')
plt.ylabel('MSE')
plt.show()

