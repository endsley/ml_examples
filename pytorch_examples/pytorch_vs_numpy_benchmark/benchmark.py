#!/usr/bin/env python

import torch									
from torch.autograd import Variable
import numpy as np
import time


#	Test to see if numpy matrix dot is faster or pytorch mm function


A = np.random.randn(10000,10000)
A2 = torch.from_numpy(A)


start_time = time.time() 
A.dot(A)
numpy_time = (time.time() - start_time)


start_time = time.time() 
torch.mm(A2, A2)
torch_time = (time.time() - start_time)

print('Numpy time : ' ,  numpy_time)
print('Torch time : ' ,  torch_time)
