#!/usr/bin/env python

from cnn_kernel_net import *
from torch.utils.data import Dataset, DataLoader
from sklearn import linear_model
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np



db = {}
db['img_height'] = 29
db['img_width'] = 29
db['batch_size'] = 5

if torch.cuda.is_available(): db['dataType'] = torch.cuda.FloatTensor
else: db['dataType'] = torch.FloatTensor
epoc_loop = 5000
learning_rate = 1e-3
exit_loss=0.001

#imgT = transforms.Compose([transforms.ToTensor()])
#imgT = None		#transforms.ToPILImage()
imgT = transforms.ToTensor()
dset = datasets.MNIST('./data', train=True, download=True, transform=imgT )
tset = datasets.MNIST('./data', train=False, transform=imgT )

def get_loss(ckernel_net, data_loader):
	#	Compute final average loss
	loss_sum = 0
	for idx, (data, target) in enumerate(data_loader):
		data = rescale(data, db)
		dataOut = ckernel_net.CAE_forward(data)

		loss = ckernel_net.CAE_compute_loss(data)
		loss_sum += loss
	
	avgL = loss_sum/idx
	return avgL.cpu().data.numpy()[0]

def write2(epoch, v1):
		sys.stdout.write("\repoch : %d, batch : %d" % (epoch, v1))
		sys.stdout.flush()

def loss_optimization_printout(epoch, avgLoss, avgGrad, epoc_loop, slope):
		sys.stdout.write("\r\t\t\t\t%d/%d, MaxLoss : %f, AvgGra : %f, progress slope : %f" % (epoch, epoc_loop, avgLoss, avgGrad, slope))
		sys.stdout.flush()

def get_slope(y_axis):
	y_axis = np.array(list(y_axis))

	n = len(y_axis)
	LR = linear_model.LinearRegression()
	X = np.array(range(n))
	X = X.reshape((n,1))
	y_axis = y_axis.reshape((n,1))
	LR.fit(X, y_axis)
	#print LR.intercept_
	
	return LR.coef_

def rescale(data, db):

	Dnumpy = data.numpy()
	full_stack = None
	for m in range(Dnumpy.shape[0]):
		image = transform.resize(Dnumpy[m,0,:,:], (29,29), mode='constant')

		#plt.imshow(image, cmap='gray')
		#plt.show()
		#import pdb; pdb.set_trace()
	
		image = torch.from_numpy(image)
		image = image.unsqueeze(dim=0).unsqueeze(dim=0)
	
		if type(full_stack) == type(None):
			full_stack = image
		else:
			full_stack = torch.cat((full_stack, image), dim=0)

	full_stack = Variable(full_stack.type(db['dataType']))
	return full_stack



train_loader = torch.utils.data.DataLoader( dset, batch_size=5, shuffle=True)
test_loader = torch.utils.data.DataLoader(tset, batch_size=5, shuffle=True)

if torch.cuda.is_available(): ckernel_net = cnn_kernel_net(db, 5).cuda()
else: ckernel_net = cnn_kernel_net(db,5)

optimizer = ckernel_net.get_optimizer()

    

avgLoss_cue = collections.deque([], 400)
for epoch in range(epoc_loop):
	running_avg = []
	running_avg_grad = []

	for idx, (data, target) in enumerate(train_loader):
		write2(epoch, idx)
		data = rescale(data, db)
		dataOut = ckernel_net.CAE_forward(data)

		optimizer.zero_grad()
		loss = ckernel_net.CAE_compute_loss(data)
		loss.backward()
		optimizer.step()

		grad_norm = 0	
		for param in ckernel_net.parameters():
			grad_norm += param.grad.data.norm()

		running_avg_grad.append(grad_norm)
		running_avg.append(loss.data[0])

	maxLoss = np.max(np.array(running_avg))		#/db['num_of_output']
	avgGrad = np.mean(np.array(running_avg_grad))
	avgLoss_cue.append(maxLoss)
	progression_slope = get_slope(avgLoss_cue)

	loss_optimization_printout(epoch, maxLoss, avgGrad, epoc_loop, progression_slope)

	if maxLoss < exit_loss: break;
	if len(avgLoss_cue) > 50 and progression_slope > 0: break;


	#	Save result
	avgLoss = get_loss(ckernel_net, train_loader)
	print('\nEnding avg loss %.3f.'%avgLoss)
	
	
	try:
		prev_result = pickle.load( open( "mnist.p", "rb" ) )
	except:
		prev_result = {}
		prev_result['avgLoss'] = 1000000
	
	if prev_result['avgLoss'] > avgLoss:
		result = {}
		result['avgLoss'] = avgLoss
		result['kernel_net'] = ckernel_net
		pickle.dump( result, open( "mnist.p", "wb" ) )


import pdb; pdb.set_trace()


#----------------------------------------------------------------
#
#def rescale(data, db):
#
#	Dnumpy = data.numpy()
#	full_stack = None
#	for m in range(Dnumpy.shape[0]):
#		image = transform.resize(Dnumpy[m,0,:,:], (29,29), mode='constant')
#
#		#plt.imshow(image, cmap='gray')
#		#plt.show()
#		#import pdb; pdb.set_trace()
#	
#		image = torch.from_numpy(image)
#		image = image.unsqueeze(dim=0).unsqueeze(dim=0)
#	
#		if type(full_stack) == type(None):
#			full_stack = image
#		else:
#			full_stack = torch.cat((full_stack, image), dim=0)
#
#	full_stack = Variable(full_stack.type(db['dataType']))
#	return full_stack
#
#def write2(v1):
#		sys.stdout.write("\rbatch : %d" % (v1))
#		sys.stdout.flush()
#
#def get_loss(ckernel_net, data_loader):
#	#	Compute final average loss
#	loss_sum = 0
#	for idx, (data, target) in enumerate(data_loader):
#		write2(idx)
#		data = rescale(data, db)
#		loss = ckernel_net.CAE_compute_loss(data)
#		loss_sum += loss
#	
#	avgL = loss_sum/idx
#	return avgL.cpu().data.numpy()[0]
#
#def save_to_img(ckernel_net, data_loader):
#	#	Compute final average loss
#	loss_sum = 0
#
#	total_tensor = None
#	total_label = None
#	for idx, (data, target) in enumerate(data_loader):
#		write2(idx)
#		data = rescale(data, db)
#
#		if type(total_tensor) == type(None):
#			total_tensor = data
#		else:
#			total_tensor = torch.cat((total_tensor, data), dim=0)
#
#		if type(total_label) == type(None):
#			total_label = target
#		else:
#			total_label = torch.cat((total_label, target), dim=0)
#
#		#import pdb; pdb.set_trace()
#		#data = data.squeeze(dim=0).squeeze(dim=0)
#		#x = data.cpu().data.numpy()
#		#plt.imshow(x, cmap='gray')
#		#plt.savefig('imgs/' + str(idx) + '.png')
#	
#		#fin.write(str(idx) + '.png\n')
#		#fin2.write(str(target.numpy()[0]) + '\n')
#
#
#	pickle.dump( total_tensor , open( 'mnist_30_validation.pk', "wb" ) )
#	pickle.dump( total_label , open( 'mnist_30_label_validation.pk', "wb" ) )
#	import pdb; pdb.set_trace()
#		#import pdb; pdb.set_trace()	
#
#imgT = transforms.ToTensor()
#dset = datasets.MNIST('./data', train=True, download=True, transform=imgT )
#tset = datasets.MNIST('./data', train=False, transform=imgT )
#train_loader = torch.utils.data.DataLoader( dset, batch_size=5, shuffle=True)
#test_loader = torch.utils.data.DataLoader(tset, batch_size=100, shuffle=True)
#
#
#prev_result = pickle.load( open( "mnist.p", "rb" ) )
#print('Loss : %.3f'%prev_result['avgLoss'])
#ckernel_net = prev_result['kernel_net']
##outLoss = get_loss(ckernel_net, train_loader)
##print('Loss : %.3f'%outLoss)
#
#save_to_img(ckernel_net, test_loader)
#
#



##----------------------------------------------------------------
#
#perm = np.random.permutation(10000)[0:2000]
#
##perm = np.random.permutation(10)[0:3]
##A = np.empty((10,1,29,29))
##B = torch.Tensor(A)
##C = B[perm, :,:,:]
##print C.shape
##import pdb; pdb.set_trace()
#
#
#data = pickle.load( open( "mnist_20_validation.pk", "rb" ) )
#labels = pickle.load( open( "mnist_20_label_validation.pk", "rb" ) )
#
#D = data[perm,:,:,:].cpu()
#L = labels[perm].cpu()
#
#pickle.dump( D, open( "mnist_20.pk", "wb" ) )
#pickle.dump( L, open( "mnist_20_label.pk", "wb" ) )
#
#
#import pdb; pdb.set_trace()
#
#plt.imshow(D[0,0,:,:].cpu().data.numpy(), cmap='gray')
#plt.show()
#
#print L[0]
#
#import pdb; pdb.set_trace()
