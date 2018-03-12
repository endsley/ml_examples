#!/usr/bin/env python

import torch
from torch.autograd import Variable

class SGD():
	def __init__(self, model, lr):
		self.model = model
		self.lr = lr

	def zero_grad(self):
		pass

	def step(self):
		for param in self.model.parameters():
			param.data -= self.lr * param.grad.data

