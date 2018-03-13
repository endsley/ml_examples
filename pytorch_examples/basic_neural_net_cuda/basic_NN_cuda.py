#!/usr/bin/env python


#This example show how to use pytorch to solve a small neural net
#	input layer 3 dimension
#	hidden layer 4 dimension
#	output layer 1 dimension, sigmoid activation

import torch
from torch.autograd import Variable
import numpy as np



d = 3
hidden_d = 4
output_d = 1
#dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

learning_rate = 1

x = np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]])
y = np.array([[0.3],[0.3],[0.6],[0.6],[1],[1]])

x = torch.from_numpy(x)
x = Variable(x.type(dtype), requires_grad=True)		#True cus variable

y = torch.from_numpy(y)
y = Variable(y.type(dtype), requires_grad=False)	#False cus constant, if this changes, network needs to be rebuilt

#	Network structure
NN = torch.nn.Sequential(
	torch.nn.Linear(d, hidden_d, bias=True),
	torch.nn.ReLU(),
	torch.nn.Linear(hidden_d, hidden_d, bias=True),
	torch.nn.ReLU(),
	torch.nn.Linear(hidden_d, output_d, bias=True),
	torch.nn.Sigmoid(),
)
NN = NN.cuda()

for m in range(5000):
	y_estimate = NN(x)				# Forward pass
	loss = (y_estimate - y).norm()		# norm of the error as loss function
	NN.zero_grad()
	loss.backward()
	print 'm : ', m , ' , ' , 'loss : ' , 
	print loss.data.cuda()[0] , ' , learning_rate : ' , learning_rate 

	while True:		# adjust learning rate
		for param in NN.parameters():	# Gradient descent
			param.data -= learning_rate * param.grad.data

		new_y_estimate = NN(x)				
		new_loss = (new_y_estimate - y).norm()		

		if new_loss.data[0] > loss.data[0]:
			for param in NN.parameters():	# reset the changes
				param.data += learning_rate * param.grad.data

			learning_rate = learning_rate*0.5
		else:
			break

		if learning_rate < 0.0000001: break


print 'Results from the initial training'
print 'Estimate : ' , y_estimate.data.cuda()
print 'Real     : ' , y.data.cuda() , '\n'



