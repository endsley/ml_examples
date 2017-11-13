#!/usr/bin/env python

from eig_sorted import *
import time 
from CPM import *

def basic_test():
	d = 1
	A = np.random.randn(2048,2048)
	#A = A + A.T
	A = A.dot(A.T)

	#	Ran coordinate wise power method
	start_time = time.time() 
	cpm = CPM(A, Num_of_eigV=d, style='dominant_first') #largest_first, dominant_first, smallest_first, least_dominant_first
	cpm_time = (time.time() - start_time)


	#	Ran regular eig decomposition method
	start_time = time.time() 
	[V,D] = eig_sorted(A)
	eig_time = (time.time() - start_time)

#	print cpm.eigValues
#	print cpm.eigVect , '\n\n'
#
#	print V
#	print D

	print cpm.eigValues , ' , ' , D[0:d]
	print np.hstack((cpm.eigVect[0:4, 0:d], V[0:4, 0:d]))
	print 'cpm_time : ' , cpm_time , '  ,  ' , 'eig_time : ', eig_time


def very_large_matrix():
	d = 1
	A = np.random.randn(50000,50000)
	A = A + A.T

	#	Ran coordinate wise power method
	#largest_first, dominant_first, smallest_first, least_dominant_first
	start_time = time.time() 
	cpm = CPM(A, Num_of_eigV=d, style='largest_first', init_with_power_method=False, starting_percentage=0.004) 
	cpm_time = (time.time() - start_time)

	print 'cpm_time : ' , cpm_time 


#very_large_matrix()
basic_test()
