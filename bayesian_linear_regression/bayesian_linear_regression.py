#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
from numpy.linalg import inv
from numpy.random import rand, randn
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, invgamma
from scipy.special import gamma as Γ
from numpy import exp as e
from numpy import log as ln
from numpy import pi as π


#import pdb; pdb.set_trace()

#  Generate the data
n = 20000
d = 2
x = rand(n,1)
y = 3*x + 1 + 0.1*randn(n,1)  # solution w = [3,1]
X = np.hstack((x, np.ones((n,1))))

# Plot the data
plt.scatter(x, y, label='Original Signal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# In[72]:


μₒ = np.array([[0],[0]])
Σₒ = np.array([[1,0],[0,1]])
Λₒ = inv(Σₒ)
μ = inv(X.T.dot(X) + Λₒ).dot(X.T.dot(y) + Λₒ.dot(μₒ))
print(μ)

Λ = X.T.dot(X) + Λₒ
Σ = inv(Λ)

s, t = np.random.multivariate_normal(μ.flatten(), Σ, 1000).T

# Plot the points using a 2D histogram
# Notice how we have a Gaussian distribution centered around 2.96 and 1.02
plt.figure(figsize=(8, 8))
plt.hist2d(s, t, bins=30, cmap='viridis')
plt.colorbar(label='Density')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Gaussian Distribution')
plt.show()


# In[73]:


# Let's now draw the Gaussian distribution in 3d space
# Notice that the solution is centered around 
grid_size = 100
ẋ = np.linspace(0, 4, grid_size)
ỳ = np.linspace(0, 4, grid_size)
 
Ẍ, Ŷ = np.meshgrid(ẋ, ỳ)
pos = np.dstack((Ẍ, Ŷ))

# Create the Gaussian distribution
rv = multivariate_normal(μ.flatten(), Σ)
Z = rv.pdf(pos)

# Plot the 3D mesh
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Ẍ, Ŷ, Z, cmap='viridis', edgecolor='none')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Probability Density')
ax.set_title('3D Gaussian Distribution Mesh')
plt.show()


# In[82]:


a = 1
b = 1
α = n/2 + a
β = y.T.dot(y)/2 + b + μₒ.T.dot(Λₒ).dot(μₒ)/2 - μ.T.dot(Λ).dot(μ)/2

xr = np.reshape(np.linspace(0.007,0.013,100), (100,1))
yr = invgamma.pdf(xr, a=α, scale=β)

plt.plot(xr, yr)
plt.title('True σ²: 0.01, Expected σ² : %.4f'% (β/(α-1)))
plt.show()


##	Use mcmc sampling to generate the distributions

def p(ᶍ):	# the scaled probability value
	w = np.reshape(ᶍ[0:2],(2,1))
	σᒾ = ᶍ[2]

	λ = -1/(2*σᒾ)
	return (-n/2 - d/2 - a - 1)*ln(σᒾ) + λ*(y - X.dot(w)).T.dot(y - X.dot(w)) + λ*(w - μₒ).T.dot(Λₒ).dot(w - μₒ) + 2*b
	

def metropolis_sampler(N, μᵐ, Σᵐ):
	samples = np.empty((0, 3))

	while len(samples) != n:
		ᶍ = np.random.multivariate_normal(μᵐ, Σᵐ, 1)[0]
		if ᶍ[2] < 0: continue

		if rand() < p(ᶍ)/p(μᵐ): μᵐ = ᶍ
		samples = np.vstack((samples, μᵐ))
		   
	return samples

N = 2000
μᵐ = np.array([1,1,0.5])
Σᵐ = 0.3*np.eye(3)

# Generate samples
S = metropolis_sampler(N, μᵐ, Σᵐ)

import pdb; pdb.set_trace()


