#!/usr/bin/env python


from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import types
from os import listdir
from os.path import isfile, join


class image_datasets(Dataset):
	def __init__(self, root_dir, gray=True, transform=None):
		self.root_dir = root_dir
		self.transform = transform
		self.gray = gray

		self.allfiles = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
		self.image_files = []
		for i in self.allfiles:
			if i.find('.jpg') != -1 or i.find('.pgm') != -1 :
				self.image_files.append(i)

	def display_image(self, img_id):

		if type(img_id) == torch.ByteTensor:
			image = img_id
		elif type(img_id) == types.IntType:
			img_name = os.path.join(self.root_dir, self.image_files[img_id])
			image = io.imread(img_name, as_grey=self.gray)
		else:
			print("Error : unrecognized type")
			import pdb; pdb.set_trace()

		
		plt.figure()
		if self.gray: plt.imshow(image, cmap='gray')
		else: plt.imshow(image)
		plt.pause(0.001)  # pause a bit so that plots are updated
		plt.show()

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.image_files[idx])
		image = io.imread(img_name)
		image = torch.from_numpy(image)
		return image

face_data = image_datasets(root_dir='./dataset_img/')
data_loader = DataLoader(face_data, batch_size=5, shuffle=True, num_workers=4)

for i, data in enumerate(data_loader, 0):
	print(data.shape)
	face_data.display_image(data[0,:,:])
	import pdb; pdb.set_trace()	

