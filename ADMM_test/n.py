#!/usr/bin/python

import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score

x = np.array([1 ,2 ,4 ,1 ,1 ,1 ,4 ,4 ,4 ,4 ,4 ,4 ,3 ,3 ,3 ,3 ,3 ,3 ,4 ,4 ,4 ,4 ,4 ,2 ,3 ,2 ,3 ,3 ,2])
y = np.array([1, 1, 3, 2, 3, 4, 1, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 4, 2, 2, 2, 4, 2, 4, 4, 2, 2, 3])

numpy.savetxt('alternative_kernel.txt', X, fmt='%.18e', delimiter=',', newline='\n', header='', footer='')

print normalized_mutual_info_score(x, y)







