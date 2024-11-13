#!/usr/bin/env python

import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet


X = np.array([	[1,2,4],
				[2,1,0],
				[1,0,3],
				[3,4,3],
				[2,0,2],
				[6,1,7],
				[2,3,6],
				[0,3,4],
				[1,0,2],
				[1,1,1]])

X = preprocessing.scale(X)

w = np.array([[3],[2],[0]])
y = X.dot(w) + 0.2*np.random.randn(10,1)


lasso = Lasso()
lasso.fit(X,y)
ŷ = lasso.predict(X)
w = lasso.coef_
s = lasso.score(X,y)
print(w)
print(s)

ridge = Ridge().fit(X, y)
w = ridge.coef_
s = ridge.score(X,y)
print(w)
print(s)


regr = ElasticNet(random_state=0).fit(X, y)
w = ridge.coef_
s = ridge.score(X,y)
print(w)
print(s)

