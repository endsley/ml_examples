#!/usr/bin/env python

from os import listdir
from os.path import isfile, join
import numpy as np

def create_validation():
	root_dir = './'
	allfiles = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
	image_files = []
	for i in allfiles:
		if i.find('.jpg') != -1 or i.find('.pgm') != -1 or i.find('.png') != -1 :
			image_files.append(i)
	
	fn = {}
	for i in image_files:
		key = i.split('_')[0]
	
		if fn.has_key(key):
			fn[key].append(i)
		else:
			fn[key] = []
			fn[key].append(i)
	
	count = 0
	file_list = []
	label_list = []
	for i,j in fn.items():
		for p in j:
			file_list.append(p)
			label_list.append(count)
		count += 1
	
	face_file = open('face_40_validation.csv','w')
	face_label_file = open('face_40_label_validation.csv','w')
	
	face_test_file = open('face_40.csv','w')
	face_test_label_file = open('face_40_label.csv','w')

	vs = zip(label_list, file_list)
	for s,t in vs:
		face_label_file.write(str(s)+'\n')
		face_file.write(t+'\n')

		if np.random.rand() > 0.6:
			face_test_label_file.write(str(s)+'\n')
			face_test_file.write(t+'\n')


	face_file.close()
	face_label_file.close()
	face_file.close()
	face_label_file.close()
	import pdb; pdb.set_trace()

create_validation()
