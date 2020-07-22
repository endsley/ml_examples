#!/usr/bin/env python

import numpy as np
import sys
import sklearn.metrics



np.set_printoptions(precision=4)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)


dim = 100
within_variance = 1
between_variance = 10
num_samples = 300

A = np.random.randn(dim,dim)
A = A.dot(A.T)
[D,V] = np.linalg.eigh(A)
center_point_g1 = V[:,-1] 
g1_points = np.empty((0, dim))
g2_points = np.empty((0, dim))
all_points = np.empty((0, dim))

for i in range(num_samples):
	while True:
		B = A + within_variance*np.random.randn(dim,dim)
		[D,V] = np.linalg.eigh(B)

		if g1_points.shape[0] == 0:
			g1_points = np.vstack((g1_points, V[:,-1]))
			break
		else:
			U = np.reshape(V[:,-1], (dim,1))
			if np.all(g1_points.dot(U) > 0):
				g1_points = np.vstack((g1_points, V[:,-1]))
				break

#	Picking a new group center
while True:
	C = A + between_variance*np.random.randn(dim,dim)
	[D,V] = np.linalg.eigh(C)
	U = np.reshape(V[:,-1], (dim,1))

	if np.all(g1_points.dot(U) > 0):
		center_point_g2 = V[:,-1] 
		break
	
all_points = g1_points
for i in range(num_samples):
	while True:
		B = C + within_variance*np.random.randn(dim,dim)
		[D,V] = np.linalg.eigh(B)

		if g2_points.shape[0] == 0:
			g2_points = np.vstack((g2_points, V[:,-1]))
			all_points = np.vstack((all_points, V[:,-1]))
			break
		else:
			U = np.reshape(V[:,-1], (dim,1))
			if np.all(all_points.dot(U) > 0):
				g2_points = np.vstack((g2_points, V[:,-1]))
				all_points = np.vstack((all_points, V[:,-1]))
				break


X = g1_points
Y = g2_points

Cₓ= X - np.mean(X,0)
Cᵧ= Y - np.mean(Y,0)

ղ = Cₓ.shape[0]
ṁ = Cᵧ.shape[0]

Ưₓ= np.reshape(np.mean(X,0), (dim,1))
Ưᵧ= np.reshape(np.mean(Y,0), (dim,1))

Aeq = 2*ղ*Cₓ.T.dot(Cₓ) + 2*ṁ*Cᵧ.T.dot(Cᵧ)
Aneq = ṁ*(X.T - Ưᵧ).dot(X) + ղ*(Y.T - Ưₓ).dot(Y)

Tr_Aeq = np.trace(Aeq)
Tr_Aneq = np.trace(Aneq)

[γ,V] = np.linalg.eigh(Aeq)
[λ,U] = np.linalg.eigh(Aneq)

Between_d = np.linalg.norm(Ưₓ - Ưᵧ)
Aⵐ_Dia = np.max(sklearn.metrics.pairwise.pairwise_distances(all_points))
g1_Dia = np.max(sklearn.metrics.pairwise.pairwise_distances(g1_points))
g2_Dia = np.max(sklearn.metrics.pairwise.pairwise_distances(g2_points))
Aᆖ_Dia = np.max([g1_Dia, g2_Dia])


print('{:15}{:10}{:10}{:10}{:10}{:10}{:10}{:10}'.format('Num Samples', 'Aⵐ_Dia','Aᆖ_Dia', 'g1_Dia', 'g2_Dia', 'Dist Ư', 'Tr(A=)', 'Tr(Aⵐ)\n'))
print('{:<15d}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}'.format(num_samples, Aⵐ_Dia, Aᆖ_Dia, g1_Dia, g2_Dia, Between_d, Tr_Aeq, Tr_Aneq))

