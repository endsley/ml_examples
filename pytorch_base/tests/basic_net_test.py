#!/usr/bin/env python

from basic_neural_net import *
from test_base import *
from my_dataset import *


class basic_net_test(test_base):
	def __init__(self):
		db = {}
		db['data_file_name'] = '../dataset/data_4.csv'
		db['label_file_name'] = '../dataset/data_4_label.csv'
		db['model'] = basic_neural_net()

		test_base.__init__(self, db)

