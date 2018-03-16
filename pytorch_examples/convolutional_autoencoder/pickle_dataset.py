#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import pickle
import sklearn.metrics


class pickle_dataset(Dataset):
	def __init__(self, db):
		self.db = db

		self.x = pickle.load( open( db['data_file_name'], "rb" ) ).data
		self.y = pickle.load( open( db['label_file_name'], "rb" ) )

	def get_label(self):
		if self.mode == 'validation':
			return self.y_valid.numpy()
		elif self.mode == 'training':
			return self.y.numpy()
		else:
			print('Error unknown mode in dataset : %s'%self.mode)
			import pdb; pdb.set_trace()	

	def get_data(self):
		return Variable(self.x_valid.type(self.db['dataType']), requires_grad=False)

	def __getitem__(self, index):
		return self.x[index], self.y[index], index

	def __len__(self):
		return self.x.shape[0]




