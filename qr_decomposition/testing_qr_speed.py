#!/usr/bin/env python

import numpy as np
import timeit

#Testing qr run time on squared vs tall matrix
#Huge speed difference if matrix is tall

A = np.random.randn(4000, 4000)
B = np.random.randn(4000, 20)


start = timeit.default_timer()
[Q,R] = np.linalg.qr(A)
stop = timeit.default_timer()
print('A Time: ', stop - start)  

start = timeit.default_timer()
[Q,R] = np.linalg.qr(B)
stop = timeit.default_timer()
print('B Time: ', stop - start)  
