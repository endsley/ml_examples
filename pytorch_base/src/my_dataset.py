#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader



class my_dataset(Dataset):
    def __init__(self, db):
        self.x = np.loadtxt(db['data_file_name'], delimiter=',', dtype=np.float32)
        self.y = np.loadtxt(db['label_file_name'], delimiter=',', dtype=np.int32)

        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len





