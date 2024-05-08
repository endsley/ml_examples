#!/usr/bin/env python


from scipy.integrate import quad 
  
def f(x): 
  return 2*x
  
A, err = quad(f, 0, 2) 
print('Area = %.2f'%A) 
