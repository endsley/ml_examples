#!/usr/bin/env python

import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([	[0,1,1],
				[1,0,1],
				[3,2,1],
				[3,3,1]])

y = np.array(	[0,
				 0,
				 1,
				 1])

clf = LogisticRegression(random_state=0).fit(X, y)
print(clf.coef_)
print(clf.predict(X))

