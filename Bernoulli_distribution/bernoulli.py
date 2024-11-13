#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#	You know p(x) once you know θ
#	$$p(x) = \theta^x (1 - \theta)^{1 - x} $$

X = pd.read_csv("Prob_of_breakup_within_1_month.csv", header=None)
X = LabelEncoder().fit_transform(X[0])
θ = np.sum(X)/len(X)
print('θ = %.3f'%θ)

