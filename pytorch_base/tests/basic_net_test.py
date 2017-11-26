#!/usr/bin/env python

from basic_neural_net import *
from test_base import *
from my_dataset import *


class basic_net_test(test_base):
	def __init__(self):
		db = {}
		db['data_file_name'] = '../dataset/data_4.csv'
		db['label_file_name'] = '../dataset/data_4_label.csv'
		db['epoc_loop'] = 1000			#	How many time to repeat the epoch
		db['batch_size'] = 4			#	Size for each batch
		db['learning_rate'] = 0.005	
		db['print_loss'] = True
		db['dataType'] = torch.FloatTensor
		db['model'] = basic_neural_net()

		test_base.__init__(self, db)


B = basic_net_test()
B.train()
print B.db['y_pred']
import pdb; pdb.set_trace()
