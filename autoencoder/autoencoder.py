#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import numpy as np

dtype = torch.FloatTensor
N, D_in, D_out = 20, 20, 4

layers = int(np.floor(np.log(D_in)/np.log(D_out)))
W = {}


layer_id = 0
for m in range(layers):
	Din = int(D_in/np.power(2.0,m))
	Dout = int(D_in/np.power(2.0,m+1))
	W[layer_id] = Variable(torch.randn(Din, Dout).type(dtype), requires_grad=True)
	layer_id += 1

W[layer_id] = Variable(torch.randn(Dout, D_out).type(dtype), requires_grad=True)
layer_id += 1
W[layer_id] = Variable(torch.randn(D_out, Dout).type(dtype), requires_grad=True)
layer_id += 1

for m in range(layers)[::-1]:
	Din = int(D_in/np.power(2.0,m+1))
	Dout = int(D_in/np.power(2.0,m))
		
	W[layer_id] = Variable(torch.randn(Din, Dout).type(dtype), requires_grad=True)
	layer_id += 1
	
for i,j in W.items():	
	print i , j.data.numpy().shape




x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)


learning_rate = 1e-6
#for t in range(500):

h = x
for i, j in W.items():
	h = h.mm(W[i])
	if i < len(W):
		h = h.clamp(min=0)

	#if i == layers: 
	#	h
	#import pdb; pdb.set_trace()

loss = (h - x).pow(2).sum()
loss.backward()


import pdb; pdb.set_trace()



  #  y_pred = x.mm(w1).clamp(min=0).mm(w2)
  #  loss = (y_pred - y).pow(2).sum()
  #  print(t, loss.data[0])
  #  loss.backward()
  #  w1.data -= learning_rate * w1.grad.data
  #  w2.data -= learning_rate * w2.grad.data

  #  # Manually zero the gradients after updating weights
  #  w1.grad.data.zero_()
  #  w2.grad.data.zero_()
