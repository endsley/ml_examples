#!/usr/bin/env python


import numpy as np
import impyute as impy		#pip install impyute
from impyute.imputation.cs import mice




n = 5
arr = np.random.uniform(high=6, size=(n, n))
print(arr)
print('\n\n')

for _ in range(3): arr[np.random.randint(n), np.random.randint(n)] = np.nan
print(arr)



imputed_training = mice(arr)
#imputed_training = impy.mean(arr)

print('\n\n')
print(imputed_training)
print('\n\n')


