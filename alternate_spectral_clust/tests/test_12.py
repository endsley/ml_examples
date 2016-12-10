#!/usr/bin/python

import sys
sys.path.append('./lib')
from alt_spectral_clust import *
import numpy as np
#from io import StringIO   # StringIO behaves like a file object
from numpy import genfromtxt
import numpy.matlib
from sklearn.metrics.cluster import normalized_mutual_info_score
import pickle
import sklearn
import time 
from cost_function import *
import matplotlib.pyplot as plt


#data = genfromtxt('data_sets/facial_960.csv', delimiter=',')		
data = genfromtxt('data_sets/facial.csv', delimiter=',')		
label = genfromtxt('data_sets/facial_true_labels.csv', delimiter=',')		
sunglass_label = genfromtxt('data_sets/facial_sunglasses_labels.csv', delimiter=',')		
original_Y = genfromtxt('data_sets/facial_original_Y.csv', delimiter=',')		
name_file = open('data_sets/facial_names.csv', 'r')

names = np.array(name_file.readlines())
name_file.close()


ASC = alt_spectral_clust(data)
db = ASC.db

if False:	# run original spectral clustering
	ASC.set_values('q',10)
	ASC.set_values('C_num',4)
	ASC.set_values('sigma',20)
	ASC.set_values('kernel_type','Gaussian Kernel')
	ASC.run()
	original = db['allocation']
	print original

if True: 		# run preset original clustering
	db['Y_matrix'] = original_Y
	db['U_matrix'] = original_Y
	db['prev_clust'] = 1
	ASC.set_values('W_matrix',np.identity(db['d']))


if True:	# run alternative clustering
	ASC.set_values('q',10)
	ASC.set_values('sigma',80)
	ASC.set_values('C_num',4)
	ASC.set_values('lambda',2)
	start_time = time.time() 
	ASC.run()
	print("--- %s seconds ---" % (time.time() - start_time))
	alternative = db['allocation']	
	print "Alternative aginst original: " , normalized_mutual_info_score(label, alternative)
	

##print "NMI Against Sunglasses label : " , normalized_mutual_info_score(sunglass_label,alternative)
#
#print names[alternative == 1].shape
#print names[alternative == 2].shape
#print names[alternative == 3].shape
#print names[alternative == 4].shape
#
##for m in names[original== 1]: print m, 
for m in names[alternative == 1]: print m, 
##for m in names[alternative == 2]: print m, 



if False:	# save or load db to and from a pickle file
	plot_info = {}
	plot_info['debug_costVal'] = db['debug_costVal']
	plot_info['debug_gradient'] = db['debug_gradient']
	plot_info['debug_debug_Wchange'] = db['debug_debug_Wchange']
	pickle.dump( plot_info, open( "tmp_db.pk", "wb" ) )


	#plot_info= pickle.load( open( "tmp_db.pk", "rb" ) )
	#db['debug_costVal'] = plot_info['debug_costVal']
	#db['debug_gradient'] = plot_info['debug_gradient']
	#db['debug_debug_Wchange'] = plot_info['debug_debug_Wchange']



if True:	# plot the W convergence results
	X = db['data']
	plt.figure(2)
	
	plt.suptitle('facial.csv',fontsize=24)
	plt.subplot(311)
	inc = 0
	for costs in db['debug_costVal']: 
		xAxis = np.array(range(len(costs))) + inc; 
		inc = np.amax(xAxis)
		plt.plot(xAxis, costs, 'b')
		plt.plot(inc, costs[-1], 'bo', markersize=10)
		plt.title('Cost vs w iteration, each dot is U update')
		plt.xlabel('w iteration')
		plt.ylabel('cost')
		
	plt.subplot(312)
	inc = 0
	for gradient in db['debug_gradient']: 
		xAxis = np.array(range(len(gradient))) + inc; 
		inc = np.amax(xAxis)
		plt.plot(xAxis, gradient, 'b')
		plt.plot(inc, gradient[-1], 'bo', markersize=10)
		plt.title('Gradient vs w iteration, each dot is U update')
		plt.xlabel('w iteration')
		plt.ylabel('gradient')


	plt.subplot(313)
	inc = 0
	for wchange in db['debug_debug_Wchange']: 
		xAxis = np.array(range(len(wchange))) + inc; 
		inc = np.amax(xAxis)
		plt.plot(xAxis, wchange, 'b')
		plt.plot(inc, wchange[-1], 'bo', markersize=10)
		plt.title('|w_old - w_new|/|w| vs w iteration, each dot is U update')
		plt.xlabel('w iteration')
		plt.ylabel('|w_old - w_new|/|w| ')

	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
	plt.subplots_adjust(top=0.85)
	plt.show()




import pdb; pdb.set_trace()
