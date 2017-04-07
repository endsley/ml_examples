#!/usr/bin/python

import sys
sys.path.append('./lib')
from alt_spectral_clust import *
import numpy as np
from numpy import genfromtxt
import numpy.matlib
from sklearn.metrics.cluster import normalized_mutual_info_score
import pickle
import sklearn
import time 
from cost_function import *
import matplotlib.pyplot as plt
from Y_2_allocation import *



university_labels  = genfromtxt('data_sets/webkb_processed/webkbRaw_label_univ.csv', delimiter=',')
topic_labels  = genfromtxt('data_sets/webkb_processed/webkbRaw_label_topic.csv', delimiter=',')
data = genfromtxt('data_sets/webkb_processed/min_words.csv', delimiter=',')

d_matrix = sklearn.metrics.pairwise.pairwise_distances(data, Y=None, metric='euclidean')
sigma_value = np.median(d_matrix)

ASC = alt_spectral_clust(data)
db = ASC.db
db['original_labels'] = university_labels
db['alternate_labels'] = topic_labels

if False:
	ASC.set_values('q',4)
	ASC.set_values('C_num',4)
	ASC.set_values('sigma',sigma_value)
	ASC.set_values('kernel_type','Gaussian Kernel')
	start_time = time.time() 
	ASC.run()
	a = db['allocation']
	print "University NMI : " , normalized_mutual_info_score(a,university_labels)
	print "topic NMI : " , normalized_mutual_info_score(a, topic_labels)
	print("Time for spectral Clustering : %s seconds ---" % (time.time() - start_time))
else:
	print ':::::   USE PRE-DEFINED CLUSTERING :::::::::\n\n'
	ASC.set_values('kernel_type','Gaussian Kernel')

	db['Y_matrix'] = Allocation_2_Y(university_labels)
	db['prev_clust'] = 1
	db['allocation'] = university_labels


if True:	# run alternative clustering
	const = 1.0
	#print '--------- > Const : ', const
	#rand_lambda = 3*np.random.random()
	rand_lambda = 0.057 # this one works with FKDAC
	#rand_lambda = 1  # this work works with KDAC

	ASC.set_values('q',4)
	ASC.set_values('C_num',4)
	ASC.set_values('sigma',sigma_value)
	ASC.set_values('lambda',rand_lambda)
	#ASC.set_values('kernel_type','Linear Kernel')
	ASC.set_values('kernel_type','Gaussian Kernel')
	start_time = time.time() 
	ASC.run()
	print("--- Time took : %s seconds ---" % (time.time() - start_time))
	alternative = db['allocation']	
	against_alternative = normalized_mutual_info_score(university_labels, alternative)
	against_truth = normalized_mutual_info_score(topic_labels, alternative)

	print against_alternative
	print against_truth


#	alternative_Y = Allocation_2_Y(alternative)
#	topic_Y = Allocation_2_Y(topic_labels)
#
#
#	against_truth = normalized_mutual_info_score(topic_labels, alternative)
#	against_alternative = normalized_mutual_info_score(university_labels, alternative)
#
#	alternative_HSIC = HSIC_rbf(data, alternative_Y, sigma_value)
#	pose_HSIC = HSIC_rbf(data, topic_Y, sigma_value)
#
#
#
#	random_HSIC = np.array([])
#	for m in range(10):
#		random_allocation = np.floor(alternative_Y.shape[1]*np.random.random(data.shape[0]))
#		random_Y = Allocation_2_Y(random_allocation)
#		H = HSIC_rbf(data, random_Y, sigma_value)
#		random_HSIC = np.hstack((random_HSIC,H))
#
#	mean_RHSIC = np.mean(random_HSIC)
#	percent_diff = (alternative_HSIC - mean_RHSIC)/alternative_HSIC
#	pose_percent_diff = (pose_HSIC - mean_RHSIC)/pose_HSIC
#
#
#	txt = '\tLabel against alternative : ' + str(against_alternative) + '\n'
#	txt += '\tLabel against truth : ' + str(against_truth) + '\n'
#	txt += '\tAlternative HSIC % diff from random : ' + str(percent_diff) + '\n'
#	txt += '\tPose HSIC % diff from random : ' + str(pose_percent_diff) + '\n'
#	txt += '\tLambda used : ' + str(rand_lambda) + '\n'
#	txt += '\tLowest Cost : ' + str(db['lowest_cost']) + '\n'
#
#	print txt
#
#	#append_txt('./output.txt', txt)
#
#	#print "Alternative aginst original: " , normalized_mutual_info_score(pose_label, alternative)
#	#print "Alternative aginst truth : " , normalized_mutual_info_score(label, alternative)
	





import pdb; pdb.set_trace()
