#!/usr/bin/env python


from scipy.integrate import quad 
  
def f(x): 
  return x*x - 3*x + 4
  
I, err = quad(f, 0, 2) 
print(I) 
print(err)
