#!/usr/bin/env python

from basic_neural_net import *
from test_base import *
from my_dataset import *


class basic_net_test(test_base):
	def __init__(self):
		db = {}
		db['data_file_name'] =  '../../dataset/data_4.csv'
		db['label_file_name'] = '../../dataset/data_4_label.csv'
		db['epoc_loop'] = 4000			#	How many time to repeat the epoch
		db['batch_size'] = 5		#	Size for each batch
		db['learning_rate'] = 0.001
		db['print_loss'] = True
		using_cuda = False

		if using_cuda:
			db['dataType'] = torch.cuda.FloatTensor # or torch.FloatTensor on CPU
			db['model'] = basic_neural_net(2, 50).cuda()	#	num_of_input, num_of_hidden
		else:
			db['dataType'] = torch.FloatTensor # or torch.FloatTensor on CPU
			db['model'] = basic_neural_net(2, 50)	#	num_of_input, num_of_hidden


		test_base.__init__(self, db)


B = basic_net_test()

B.train()
print B.db['y_pred']
import pdb; pdb.set_trace()
