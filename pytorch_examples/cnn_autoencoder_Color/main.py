#!/usr/bin/env python

# Numpy
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Torchvision
import torchvision
import torchvision.transforms as transforms

# Matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt

# OS
import os
import argparse

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
	torch.cuda.manual_seed(SEED)


def print_model(encoder, decoder):
	print("============== Encoder ==============")
	print(encoder)
	print("============== Decoder ==============")
	print(decoder)
	print("")


def create_model():
	autoencoder = Autoencoder()
	print_model(autoencoder.encoder, autoencoder.decoder)
	if torch.cuda.is_available():
		autoencoder = autoencoder.cuda()
		print("Model moved to GPU in order to speed up training.")
	return autoencoder


def get_torch_vars(x):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x)

def imshow(img):
	npimg = img.cpu().numpy()
	plt.axis('off')
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()


class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()
		# Input size: [batch, 3, 32, 32]
		# Output size: [batch, 3, 32, 32]
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 12, 4, stride=2, padding=1),			# [batch, 12, 16, 16]
			nn.ReLU(),
			nn.Conv2d(12, 24, 4, stride=2, padding=1),		   # [batch, 24, 8, 8]
			nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),		   # [batch, 48, 4, 4]
			nn.ReLU(),
			#nn.Linear()
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),		   # [batch, 96, 2, 2]
#			 nn.ReLU(),
		)
		self.decoder = nn.Sequential(
#			 nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#			 nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
			nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
			nn.ReLU(),
			nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
			nn.Sigmoid(),
		)

		self.latent_image_len = 512


		self.bottleneck = nn.Linear(48*4*4, self.latent_image_len)
		self.bottleneck_out = nn.Linear(self.latent_image_len, 48*4*4)

	def forward(self, x):
		#encoded = self.encoder(x)
		#decoded = self.decoder(encoded)
		#return encoded, decoded


		encoded = self.encoder(x)
		conv_shape = encoded.shape
		encoded_flat = torch.flatten(encoded, 1)
		self.latent = self.bottleneck(encoded_flat)
		encoded3 = self.bottleneck_out(self.latent).reshape(conv_shape)

		decoded = self.decoder(encoded3)
		return encoded3, decoded

def main():
	parser = argparse.ArgumentParser(description="Train Autoencoder")
	parser.add_argument("--valid", action="store_true", default=False, help="Perform validation only.")
	parser.add_argument("--outbottleneck", action="store_true", default=False, help="output bottleneck only.")

	args = parser.parse_args()

	# Create model
	autoencoder = create_model()

	# Load data
	transform = transforms.Compose(
		[transforms.ToTensor(), ])
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=10000, shuffle=True, num_workers=2)
	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=2)
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	
	if args.outbottleneck:
		print("Outputing Bottleneck...")
		autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))

		#all_images = np.empty((0, autoencoder.latent_image_len))
		#all_labels = []

		#for i, (inputs, label) in enumerate(trainloader, 0):
		for i, (inputs, label) in enumerate(testloader, 0):
			name = './data/cfar_' + str(5) + '.csv'
			nameL = './data/cfar_' + str(5) + '_label.csv'
			
			img_batch = get_torch_vars(inputs)
			decoded_imgs = autoencoder(img_batch)[1]
			new_img_batch = autoencoder.latent.cpu().detach().numpy()

			np.savetxt(name, new_img_batch, delimiter=',', fmt='%.5f') 
			np.savetxt(nameL, label, delimiter=',', fmt='%d') 

			#all_images = np.vstack((all_images, new_img_batch))
			#all_labels.extend(label)

			#print(all_images.shape, len(all_labels))
			print(i)

			#if all_images.shape[0] >= 15792:
			#	import pdb; pdb.set_trace()

			#if i == 20:
			#	print('GroundTruth: ', ' '.join('%5s' % classes[label[j]] for j in range(16)))
			#	imshow(torchvision.utils.make_grid(decoded_imgs.data))

		exit(0)


	if args.valid:
		print("Loading checkpoint...")
		autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
		dataiter = iter(testloader)
		images, labels = dataiter.next()
		print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
		imshow(torchvision.utils.make_grid(images))

		images = Variable(images.cuda())

		decoded_imgs = autoencoder(images)[1]
		#print(autoencoder.latent)
		#import pdb; pdb.set_trace()
		imshow(torchvision.utils.make_grid(decoded_imgs.data))

		exit(0)

	# Define an optimizer and criterion
	criterion = nn.BCELoss()
	optimizer = optim.Adam(autoencoder.parameters())

	for epoch in range(100):
		running_loss = 0.0
		for i, (inputs, _) in enumerate(trainloader, 0):
			inputs = get_torch_vars(inputs)

			# ============ Forward ============
			encoded, outputs = autoencoder(inputs)
			loss = criterion(outputs, inputs)
			# ============ Backward ============
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# ============ Logging ============
			running_loss += loss.data
			if i % 2000 == 1999:
				print('[%d, %5d] loss: %.3f' %
					  (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	print('Finished Training')
	print('Saving Model...')
	if not os.path.exists('./weights'):
		os.mkdir('./weights')
	torch.save(autoencoder.state_dict(), "./weights/autoencoder.pkl")


if __name__ == '__main__':
	main()
