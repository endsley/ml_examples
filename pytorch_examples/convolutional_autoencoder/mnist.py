#!/usr/bin/env python

from cnn_kernel_net import *
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn import linear_model
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from PIL import Image




db = {}
db['img_height'] = 29
db['img_width'] = 29
db['batch_size'] = 20

np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)

#if torch.cuda.is_available(): db['dataType'] = torch.cuda.FloatTensor
#else: db['dataType'] = torch.FloatTensor
db['dataType'] = torch.FloatTensor


print db['dataType']
epoc_loop = 5000
learning_rate = 1e-3
exit_loss=0.001

def save_to_img(ckernel_net, data_loader):
	#	Compute final average loss
	loss_sum = 0

	total_tensor = None
	total_label = None
	for idx, (data, target) in enumerate(data_loader):
		write2(idx)
		data = rescale(data, db)

		if type(total_tensor) == type(None):
			total_tensor = data
		else:
			total_tensor = torch.cat((total_tensor, data), dim=0)

		if type(total_label) == type(None):
			total_label = target
		else:
			total_label = torch.cat((total_label, target), dim=0)

		#import pdb; pdb.set_trace()
		#data = data.squeeze(dim=0).squeeze(dim=0)
		#x = data.cpu().data.numpy()
		#plt.imshow(x, cmap='gray')
		#plt.savefig('imgs/' + str(idx) + '.png')
	
		#fin.write(str(idx) + '.png\n')
		#fin2.write(str(target.numpy()[0]) + '\n')


	pickle.dump( total_tensor , open( 'mnist_30_validation.pk', "wb" ) )
	pickle.dump( total_label , open( 'mnist_30_label_validation.pk', "wb" ) )
	import pdb; pdb.set_trace()
		#import pdb; pdb.set_trace()	

def get_loss(ckernel_net, data_loader):
	#	Compute final average loss
	for idx, (data, target) in enumerate(data_loader):
		data = Variable(data.type(db['dataType']))
		loss = ckernel_net.CAE_compute_loss(data)


	dataOut = ckernel_net(data)
	dataOut = dataOut.cpu().data.numpy()

	allocation = KMeans(10).fit_predict(dataOut)
	nmi = normalized_mutual_info_score(allocation, target.numpy())
	return [loss.cpu().data.numpy()[0], nmi]

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

def save_result(ckernel_net, test_loader):
	#	Save result
	[avgLoss, nmi] = get_loss(ckernel_net, test_loader)
	print('\nEnding avg loss : %.3f, nmi : %.3f'%(avgLoss, nmi))
			
	try:
		prev_result = pickle.load( open( "mnist_loss.p", "rb" ) )
	except:
		prev_result = {}
		prev_result['avgLoss'] = 1000000
	
	if prev_result['avgLoss'] > avgLoss:
		result = {}
		result['avgLoss'] = avgLoss
		result['nmi'] = nmi
		result['kernel_net'] = ckernel_net
		pickle.dump( result, open( "mnist_loss.p", "wb" ) )


	try:
		prev_result = pickle.load( open( "mnist_nmi.p", "rb" ) )
	except:
		prev_result = {}
		prev_result['nmi'] = 0
	
	if prev_result['nmi'] < nmi:
		result = {}
		result['avgLoss'] = avgLoss
		result['nmi'] = nmi
		result['kernel_net'] = ckernel_net
		pickle.dump( result, open( "mnist_nmi.p", "wb" ) )


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

def plot_img(img):
	img = (img + 1)*128.0
	img = img.astype('uint8')
	img = Image.fromarray(img,'L') 
	img.show()

def output_60000_validation():
	imgT = transforms.ToTensor()
	tset = datasets.MNIST('./data', train=True, transform=imgT )
	test_loader = torch.utils.data.DataLoader(tset, batch_size=len(tset), shuffle=True)
	
	for idx, (data, target) in enumerate(test_loader):
		data = rescale(data, db)

	
	data = data.data.cpu()
	print(type(data))
	print(data.shape)
	pickle.dump( data, open( "mnist_60000_validation.pk", "wb" ) )
	pickle.dump( target.numpy(), open( "mnist_60000_label_validation.pk", "wb" ) )


def output_10000_validation():
	imgT = transforms.ToTensor()
	tset = datasets.MNIST('./data', train=False, transform=imgT )
	test_loader = torch.utils.data.DataLoader(tset, batch_size=len(tset), shuffle=True)
	
	for idx, (data, target) in enumerate(test_loader):
		data = rescale(data, db)

	
	data = data.data.cpu()
	pickle.dump( data, open( "mnist_10000_validation.pk", "wb" ) )
	pickle.dump( target.numpy(), open( "mnist_10000_label_validation.pk", "wb" ) )
	#labels = pickle.load( open( "mnist_20_label_validation.pk", "rb" ) )


def checkout_CAE():
	X = pickle.load( open( 'mnist_60000_validation.pk', "rb" ) )
	Y = pickle.load( open( 'mnist_60000_label_validation.pk', "rb" ) )
	Y = torch.from_numpy(Y)

	kinfo = pickle.load( open( 'kernel_mnist.p', "rb" ) )
	cnn = kinfo['kernel_net']
	X_var = Variable(X.type(db['dataType']))

	xout = cnn(X_var)
	xout = xout.cpu().data.numpy()

	allocation = KMeans(10).fit_predict(xout)
	nmi = normalized_mutual_info_score(allocation, Y.numpy())
	print('nmi : %.3f', nmi)


def train_CAE():
	#imgT = transforms.Compose([transforms.ToTensor()])
	#imgT = None		#transforms.ToPILImage()
	#imgT = transforms.ToTensor()
	#dset = datasets.MNIST('./data', train=True, download=True, transform=imgT )
	#tset = datasets.MNIST('./data', train=False, transform=imgT )
	#train_loader = torch.utils.data.DataLoader( dset, batch_size=db['batch_size'], shuffle=True)
	#test_loader = torch.utils.data.DataLoader(tset, batch_size=db['batch_size'], shuffle=True)	
	
	X = pickle.load( open( 'mnist_10000_validation.pk', "rb" ) )
	Y = pickle.load( open( 'mnist_10000_label_validation.pk', "rb" ) )
	Y = torch.from_numpy(Y)



	dset = TensorDataset(X,Y)
	test_loader = torch.utils.data.DataLoader(dset, batch_size=db['batch_size'], shuffle=True)
	outputloader = torch.utils.data.DataLoader(dset, batch_size=len(dset), shuffle=True)
	
	#if torch.cuda.is_available(): ckernel_net = cnn_kernel_net(db, 5).cuda()
	#else: ckernel_net = cnn_kernel_net(db,5)
	ckernel_net = cnn_kernel_net(db,5)

	optimizer = ckernel_net.get_optimizer()
	   
	
	avgLoss_cue = collections.deque([], 100)
	for epoch in range(epoc_loop):
		running_avg = []
		running_avg_grad = []
	
		for idx, (data, target) in enumerate(test_loader):
			write2(epoch, idx)
			data = Variable(data.type(db['dataType']))

			optimizer.zero_grad()
			loss = ckernel_net.CAE_compute_loss(data)
			loss.backward()
			optimizer.step()
	
			grad_norm = 0	
			for param in ckernel_net.parameters():
				#print param
				#import pdb; pdb.set_trace()
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
	
			#data = rescale(data, db)
			#dataOut = ckernel_net.CAE_forward(data)

		save_result(ckernel_net, outputloader)
	
	import pdb; pdb.set_trace()

#output_60000_validation()
#output_10000_validation()
#train_CAE()
checkout_CAE()


#----------------------------------------------------------------
#
#
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




