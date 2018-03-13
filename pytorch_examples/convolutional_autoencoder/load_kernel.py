#!/usr/bin/env python

import sys
sys.path.append('../pytorch_load_image/')
import pickle
import torch
import numpy as np
from img_load import *
from cnn_kernel_net import *
from torch.autograd import Variable
import torch.nn.functional as F


face_data = image_datasets('../../dataset/faces/', 'face_img')
data_loader = DataLoader(face_data, batch_size=len(face_data), shuffle=True, drop_last=True)
result = pickle.load( open( "face.p", "rb" ) )

for idx, data in enumerate(data_loader):
	print data.shape

import pdb; pdb.set_trace()


result['avgLoss']
result['kernel_net']

