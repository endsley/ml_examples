#!/usr/bin/env python

import numpy as np
from numpy import ones
from numpy import array
import matplotlib.pyplot as plt
from numpy import mean

#	Given the Objective 
#	$$ \min_{\alpha, \beta} \; 5\alpha^2 + 6 \alpha \beta - 16 \alpha + 3 \beta^2 - 12 \beta + 14$$

#	The derivative is 
#	$$ \begin{cases} \frac{df}{d\alpha} = 10\alpha + 6\beta - 16 \\  \frac{df}{d\alpha} = 6\alpha + 6\beta - 12  \end{cases} = \begin{bmatrix} 10 & 6  \\ 6 & 6 \end{bmatrix} \begin{bmatrix} \alpha \\ \beta \end{bmatrix} - \begin{bmatrix} 16 \\ 12 \end{bmatrix} = A w - c$$
#	Let 

A = array([[10,6],[6,6]])
c = array([[16],[12]])
w = array([[2],[2]]) # solution is 1,1
η = 0.1

def f(w):
	α = w[0]
	β = w[1]
	
	return 5*α*α + 6*α*β - 16*α + 3*β*β - 12*β + 14

f_value_list = []
for i in range(20):
	fᑊ = A.dot(w) - c
	w = w - η*fᑊ				# gradient descent update w
	f_value_list.append(f(w))

print('Best w = \n', w)
plt.plot(f_value_list)
plt.title('MSE over GD')
plt.xlabel('steps')
plt.ylabel('MSE')


# Display the plot
plt.show()
