#!/usr/bin/env python

import numpy as np
import sys
from scipy.special import softmax

np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=400)
np.set_printoptions(suppress=True)

def gumbel(logit):
	τ = 0.1
	logit = np.atleast_2d(logit)
	R = np.random.rand(logit.shape[0], logit.shape[1])
	ε = -np.log(-np.log(R))
	
	noisy_logits = (ε + logit) / τ
	C = softmax(noisy_logits, axis=1)
	return C

X = np.random.rand(8,3)
C = gumbel(X)

print(X,'\n')
print(C)
