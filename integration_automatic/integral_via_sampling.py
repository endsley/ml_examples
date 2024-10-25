#!/usr/bin/env python

import numpy as np
from numpy import mean
from scipy.integrate import quad 
  
# This example shows that


#	$$ \int_0^1 \; (x+1) p(x) \; dx \approx \frac{1}{n} \sum_i \; (x+1) p(x) \quad \text{where} \quad p(x) = x^2 $$


def f(x): 
  return (x+1)*x*x
  
A, err = quad(f, 0, 1) 
print('Area = %.2f'%A) 


# via sampling
X = np.random.rand(10000)
print('Area via sampling= %.2f'%mean(f(X))) 

