#!/usr/bin/python

#	This is the main file to run various tests
#	To run a particular test, simply uncomment the import
#	while comment all other import tests



import sys
sys.path.append('./tests')
try: import numpy as np
except: 'numpy is missing from the python library, please import it first'
try: import matplotlib
except: 'matplotlib is missing from the python library, please import it first'
try: import sklearn
except: 'sklearn is missing from the python library, please import it first'
try: import scipy
except: 'scipy is missing from the python library, please import it first'

#	numpy settings
np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)





#	Tests : you run the test by importing it

#import test_1_linear_kernel
#import test_2_gaussian_kernel
#import test_3_polynomial_kernel	
#import test_4_small_gaussian
#import test_5_alternative
#import test_6_alternative
#import test_7_polynomial
#import test_7_polynomial
#import test_8
#import test_9
#import test_10
#import test_11
import test_12
#import gene_data_test
