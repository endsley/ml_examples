#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing


class my_dataset(Dataset):
	def __init__(self, db):
		self.db = db
		self.x = np.loadtxt(db['data_file_name'], delimiter=',', dtype=np.float32)
		self.x = preprocessing.scale(self.x)
		self.y = np.loadtxt(db['label_file_name'], delimiter=',', dtype=np.int32)
		self.y = self.y.reshape((self.y.shape[0],1))

		self.x = torch.from_numpy(self.x)
		self.y = torch.from_numpy(self.y)

		self.len = self.x.shape[0]

	def get_data(self):
		return Variable(self.x.type(self.db['dataType']), requires_grad=False)

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def __len__(self):
		return self.len





