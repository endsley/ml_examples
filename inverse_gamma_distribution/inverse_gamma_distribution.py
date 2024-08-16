#!/usr/bin/env python

import numpy as np
from scipy.special import gamma as Γ
from numpy import exp as e
from numpy import inf as ᨖ
from scipy.integrate import quad 



def IG(x):	#inverse gamma distribution
	a = 3
	b = 1

	return (b**a/Γ(a))*(x)**(-a-1)*e(-b/x)


area , err = quad(IG, 0, ᨖ ) 
print(area)
