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
from Y_2_allocation import *
import matplotlib 
import calc_cost
from PIL import Image
colors = matplotlib.colors.cnames
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing



np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)


im = Image.open("data_sets/Mariposa.png")
#im = Image.open("data_sets/Flower_174x175.png")
#im = Image.open("data_sets/Flower_70x70.png")

rgb_im = im.convert('RGB')
Img_3d_array = np.asarray(rgb_im)
before_preprocess_data = np.empty((0, 3), dtype=np.uint8)
data = np.empty((0, 3))
data_dic = {}
data_alloc = {}

for i in range(Img_3d_array.shape[0]):
	for j in range(Img_3d_array.shape[1]):
		data_dic[ str(Img_3d_array[i,j]) ] = Img_3d_array[i,j]


for i,j in data_dic.items():
	before_preprocess_data = np.vstack((before_preprocess_data, j))
	data = np.vstack((data, j))

print data.shape

#data = preprocessing.scale(data)
#
#
#d_matrix = sklearn.metrics.pairwise.pairwise_distances(data, Y=None, metric='euclidean')
#sigma = 0.1*np.median(d_matrix)
#
#ASC = alt_spectral_clust(data)
#db = ASC.db
#
#if True: #	Calculating the original clustering
#	ASC.set_values('q',2)
#	ASC.set_values('C_num',2)
#	ASC.set_values('sigma',sigma)
#	ASC.set_values('kernel_type','Gaussian Kernel')
#	ASC.run()
#	a = db['allocation']
#
#	for m in range(len(a)):
#		data_alloc[str(before_preprocess_data[m,:])] = a[m]
#
#if True:	# Plot the clustered image	
#	out_img = np.zeros(Img_3d_array.shape[0:2])
#	for i in range(Img_3d_array.shape[0]):
#		for j in range(Img_3d_array.shape[1]):
#			out_img[i,j] = 255*(data_alloc[str(Img_3d_array[i,j])] - 1)
#
#
#	out_img = out_img.astype('uint8')
#
#	#img = Image.fromarray(out_img, 'L') 
#	#img.save('my.png') 
#	#img.show()
#
#if False:	#	Plot clustering results
#	X = db['data']
#	fig = plt.figure()
#	ax = fig.add_subplot(111, projection='3d')
#	Uq_a = np.unique(a)
#	
#	group1 = X[a == Uq_a[0]]
#	group2 = X[a == Uq_a[1]]
#	
#	ax.scatter(group1[:,0], group1[:,1], group1[:,2], c='b', marker='o')
#	ax.scatter(group2[:,0], group2[:,1], group2[:,2], c='r', marker='x')
#	ax.set_xlabel('Feature 1')
#	ax.set_ylabel('Feature 2')
#	ax.set_zlabel('Feature 3')
#	ax.set_title('Original Clustering')
#	
#	plt.show()
#
#
######	Now we start running the alternative
#
#print ':::::   USE PRE-DEFINED CLUSTERING :::::::::\n\n'
#im = Image.open("data_sets/Flower2.png")
#rgb_im = im.convert('RGB')
#Img_3d_array = np.asarray(rgb_im)
#before_preprocess_data = np.empty((0, 3), dtype=np.uint8)
#original_allocation = np.empty((0, 1))
#data = np.empty((0, 3))
#data_dic = {}
#pixel_allocation = out_img/255
#
#
##	Compress data into identical colors, due to too many repeated colors
#for i in range(Img_3d_array.shape[0]):
#	for j in range(Img_3d_array.shape[1]):
#		data_dic[ str(Img_3d_array[i,j]) ] = [Img_3d_array[i,j], pixel_allocation[i,j]]
#
#for i,j in data_dic.items():
#	before_preprocess_data = np.vstack((before_preprocess_data, j[0]))
#	data = np.vstack((data, j[0]))
#	original_allocation = np.vstack((original_allocation, j[1]))
#
#
#data = preprocessing.scale(data)
#
#
#d_matrix = sklearn.metrics.pairwise.pairwise_distances(data, Y=None, metric='euclidean')
#sigma = 0.1*np.median(d_matrix)
#
#
#ASC_2 = alt_spectral_clust(data)
#ASC_2.set_values('kernel_type','Gaussian Kernel')
#ASC_2.db['Y_matrix'] = Allocation_2_Y(original_allocation)
#ASC_2.db['prev_clust'] = 1
#ASC_2.db['allocation'] = original_allocation
#
#if True:	# run alternative clustering
#	rand_lambda = 1.0 
#
#	ASC_2.set_values('q',2)
#	ASC_2.set_values('C_num',2)
#	ASC_2.set_values('sigma',sigma)
#	ASC_2.set_values('lambda',rand_lambda)
#	ASC_2.set_values('kernel_type','Gaussian Kernel')
#	start_time = time.time() 
#	ASC_2.run()
#	print("--- Time took : %s seconds ---" % (time.time() - start_time))
#	alternative = ASC_2.db['allocation']	
#
#
#if True:	# Plot the clustered image
#	data_alloc2 = {}
#	for m in range(len(alternative)):
#		data_alloc2[str(before_preprocess_data[m,:])] = alternative[m]
#
#	alt_img = np.zeros(Img_3d_array.shape[0:2])
#	for i in range(Img_3d_array.shape[0]):
#		for j in range(Img_3d_array.shape[1]):
#			alt_img[i,j] = 255*(data_alloc2[str(Img_3d_array[i,j])] - 1)
#
#
#	alt_img = alt_img.astype('uint8')
#	img = Image.fromarray(alt_img, 'L') 
#	img.save('alternative_img.png') 
#	img.show()
#
