#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')
import wuml
import wplotlib

#	This code reduces the dimension of the data via LDA

data = wuml.wData(xpath='../dataset/wine.csv', ypath='../dataset/wine_label.csv', label_type='discrete')

cf = wuml.classification(data, split_train_test=True, classifier='LDA') 
wuml.jupyter_print(cf.result_summary(print_out=False))
lda_X = cf.project_data_onto_linear_weights()


import pdb; pdb.set_trace()

#data = wuml.make_classification_data( n_samples=200, n_features=5, n_informative=3)
#cf = wuml.classification(data, classifier='GP')
#wuml.jupyter_print('Running a single classifier')
#wuml.jupyter_print(cf.result_summary(print_out=False))
#wuml.jupyter_print(cf)

