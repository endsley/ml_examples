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
from mpl_toolkits.mplot3d import Axes3D
from Y_2_allocation import *
from sklearn import preprocessing


fsize = '1000'
file_name = 'Four_gaussian_3D_' + fsize

data = genfromtxt('data_sets/' + file_name + '.csv', delimiter=',')		
data = preprocessing.scale(data)

label_1 = genfromtxt('data_sets/Four_gaussian_3D_' + fsize + '_original_label.csv', delimiter=',')		
label_2 = genfromtxt('data_sets/Four_gaussian_3D_' + fsize + '_alt_label.csv', delimiter=',')		

ASC = alt_spectral_clust(data)
omg = objective_magnitude
db = ASC.db

#np.savetxt('facial_sunglasses_labels.csv', sunglasses_labels, delimiter=',', fmt='%d')

if True:	#	initial clustering
	ASC.set_values('q',1)
	ASC.set_values('C_num',2)
	ASC.set_values('sigma',2)
	ASC.set_values('kernel_type','Gaussian Kernel')
	ASC.run()

	#np.savetxt('Four_gaussian_label_1.csv', db['Y_matrix'], delimiter=',', fmt='%d')
else: #	Predefining the original clustering, the following are the required settings
	ASC.set_values('q',1)
	ASC.set_values('C_num',2)
	ASC.set_values('sigma',4)
	ASC.set_values('kernel_type','Gaussian Kernel')
	ASC.set_values('W_matrix',np.identity(db['d']))

	db['Y_matrix'] = label_2
	db['U_matrix'] = label_2
	db['prev_clust'] = 1
	db['allocation'] = Y_2_allocation(label_2)
	a = db['allocation']
	b = np.concatenate((np.zeros(200), np.ones(200)))
	#print 'Predefined allocation :' , a , '\n'


if True:	#	Alterntive clustering
	ASC.set_values('sigma',1)
	ASC.set_values('W_opt_technique','DG')		# DG, SM, or ISM
	ASC.set_values('Experiment_name','Four_Gauss')
	#	only matters if picking DG
	db['DG_init_W_from_pickle'] = True
	db['pickle_count'] = 9

	start_time = time.time() 
	ASC.run()

	db['run_alternative_time'] = (time.time() - start_time)
	print("--- %s seconds ---" % db['run_alternative_time'])

	b = db['allocation']

	print "1 NMI : " , normalized_mutual_info_score(b, label_1)
	print "2 NMI : " , normalized_mutual_info_score(b, label_2)


	#np.savetxt('Four_gaussian_label_2.csv', db['Y_matrix'][:,2:4], delimiter=',', fmt='%d')


#	Output the result to a file
if True:
	cf = db['cf']
	final_cost = cf.calc_cost_function(db['W_matrix'])
	against_truth = np.round(normalized_mutual_info_score(label_2, b),3)
	against_alternative = np.round(normalized_mutual_info_score(label_1, b),3)

	outLine = str(db['N']) + '\t' + db['W_opt_technique'] + '\t' + str(np.round(db['run_alternative_time'],3)) + '\t'
	outLine += str(np.round(final_cost ,3)) + '\t' + str(np.round(cf.cluster_quality(db), 3)) + '\t'
	outLine += str(against_truth) + '\t' + str(against_alternative) + '\n' 

	fin = open('Four_gaussian_3D_' + fsize + '_' + str(db['N']) + 'x' + str(db['d']) + '_result.txt','a')
	fin.write(outLine)
	fin.close()





if False:	#	Plot clustering results
	X = db['data']
	fig = plt.figure()
	ax = fig.add_subplot(211, projection='3d')
	Uq_a = np.unique(a)
	
	group1 = X[a == Uq_a[0]]
	group2 = X[a == Uq_a[1]]
	
	ax.scatter(group1[:,0], group1[:,1], group1[:,2], c='b', marker='o')
	ax.scatter(group2[:,0], group2[:,1], group2[:,2], c='r', marker='x')
	ax.set_xlabel('Feature 1')
	ax.set_ylabel('Feature 2')
	ax.set_zlabel('Feature 3')
	ax.set_title('Original Clustering')
	
	ax = fig.add_subplot(212, projection='3d')
	Uq_b = np.unique(b)
	group1 = X[b == Uq_b[0]]
	group2 = X[b == Uq_b[1]]
	
	ax.scatter(group1[:,0], group1[:,1], group1[:,2], c='b', marker='o')
	ax.scatter(group2[:,0], group2[:,1], group2[:,2], c='r', marker='x')
	ax.set_xlabel('Feature 1')
	ax.set_ylabel('Feature 2')
	ax.set_zlabel('Feature 3')
	ax.set_title('Alternative Clustering')
	
	plt.show()


if False:	# save or load db to and from a pickle file
	plot_info = {}
	plot_info['debug_costVal'] = db['debug_costVal']
	plot_info['debug_gradient'] = db['debug_gradient']
	plot_info['debug_debug_Wchange'] = db['debug_debug_Wchange']
	pickle.dump( plot_info, open( "tmp_db.pk", "wb" ) )


	#db= pickle.load( open( "tmp_db.pk", "rb" ) )



if False:	# plot the W convergence results
	X = db['data']
	plt.figure(2)
	
	plt.suptitle('Four_gaussian_3D.csv',fontsize=24)
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


