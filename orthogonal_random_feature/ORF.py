#!/usr/bin/env python

from scipy.linalg import hadamard
import numpy as np


n = 4
H = (1/np.sqrt(n))*hadamard(n)

import pdb; pdb.set_trace()
