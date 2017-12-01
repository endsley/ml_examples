#!/usr/bin/env python

from my_dataset import *
from pytorch_base import *

class test_base():
	def __init__(self, db):
		self.db = db
		self.db['dataset'] = my_dataset(db)
		dlen = self.db['dataset'].x.shape[0]
		self.db['data_loader'] = DataLoader(dataset=self.db['dataset'], batch_size=db['batch_size'], shuffle=True)
		self.db['data_loader_full'] = DataLoader(dataset=self.db['dataset'], batch_size=dlen, shuffle=True)

		self.tbase = pytorch_base(self.db)

	def train(self):
		self.tbase.optimize()
		x = self.db['dataset'].get_data()

		self.db['y_pred'] = self.db['model']( x )
