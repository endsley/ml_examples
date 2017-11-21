#!/usr/bin/env python


class test_base():
	def __init__(self, db):
		self.db = db
		self.data = my_dataset(db)
		self.tbase = pytorch_base(db)

	#def train(self):
