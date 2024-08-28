#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
from numpy.linalg import inv
from numpy.random import rand, randn, normal
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

## Plot the data
#plt.scatter(x, y, label='Original Signal')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.legend()
#plt.show()


# In[72]:


μₒ = np.array([[0],[0]])
Σₒ = np.array([[1,0],[0,1]])
Λₒ = inv(Σₒ)
μ = inv(X.T.dot(X) + Λₒ).dot(X.T.dot(y) + Λₒ.dot(μₒ))
print(μ)

Λ = X.T.dot(X) + Λₒ
Σ = inv(Λ)

s, t = np.random.multivariate_normal(μ.flatten(), Σ, 1000).T

## Plot the points using a 2D histogram
## Notice how we have a Gaussian distribution centered around 2.96 and 1.02
#plt.figure(figsize=(8, 8))
#plt.hist2d(s, t, bins=30, cmap='viridis')
#plt.colorbar(label='Density')
#plt.xlabel('X-axis')
#plt.ylabel('Y-axis')
#plt.title('2D Gaussian Distribution')
#plt.show()


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

## Plot the 3D mesh
#fig = plt.figure(figsize=(10, 8))
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(Ẍ, Ŷ, Z, cmap='viridis', edgecolor='none')
#
#ax.set_xlabel('X-axis')
#ax.set_ylabel('Y-axis')
#ax.set_zlabel('Probability Density')
#ax.set_title('3D Gaussian Distribution Mesh')
#plt.show()



#	Generate the distribution for w and σ²
a = 1
b = 1
α = n/2 + a
β = y.T.dot(y)/2 + b + μₒ.T.dot(Λₒ).dot(μₒ)/2 - μ.T.dot(Λ).dot(μ)/2

xr = np.reshape(np.linspace(0.007,0.013,100), (100,1))
yr = invgamma.pdf(xr, a=α, scale=β)

#plt.plot(xr, yr)
#plt.title('True σ²: 0.01, Expected σ² : %.4f'% (β/(α-1)))
#plt.show()




#	Here, we simplify the joint distribution 
#	1. First sample from joint to get σᒾ
#	2. Next we sample from joint to get w given σᒾ
#	Since the joint after manipulation results in
#		inv Gamma and Gaussian, we can simply sample 
#		from these 2 distributions.
#	If not conjugate prior, we would need to use mcmc
Ẋ = invgamma.rvs(a=α, scale=β, size=n)

samples = np.empty((0, 2))
for σ2 in Ẋ:
	Λₑ = Λ/σ2
	Σₑ = inv(Λₑ)
	ᶍ = np.random.multivariate_normal(μ.flatten(), Σₑ, 1)[0]
	samples = np.vstack((samples, ᶍ))

print('E[μ] : ', np.mean(samples, axis=0))
print('E[σᒾ] : ', np.mean(Ẋ))




#	Generate the distribution for w and σ²
#	But this time, assume we didn't know the posterior
#	We will use mcmc to generate samples first from σᒾ
#	and then given σᒾ, we will again use mcmc to generate w samples


#	Remember to sample from σᒾ, we can treat everything else as a constant
#	and use the log of p(x) instead
def p1(σᒾ, w):	
	if σᒾ < 0: return 0 
	γ = 1/(2*σᒾ)
	return (-n/2 - d/2 - a - 1)*ln(σᒾ) - γ*(y - X.dot(w)).T.dot(y - X.dot(w)) - γ*(w-μₒ).T.dot(Λₒ).dot(w-μₒ) - 2*γ*b


def p2(σᒾ, w):	
	γ = 1/(2*σᒾ)
	return - γ*(y - X.dot(w)).T.dot(y - X.dot(w)) - γ*(w-μₒ).T.dot(Λₒ).dot(w-μₒ)


burn = 1000
w_samples = np.empty((0, 2))
σᒾ_samples = []
w = np.array([[2],[2]])
Σᵥ = np.array([[0.5,0],[0,0.5]])
σᒾₒ = 1
σᒾᵥ = 0

while len(σᒾ_samples) < 20000 + burn:
	#	sampling σᒾ
	σᒾᵥ = normal(σᒾₒ, 0.5) # generate a new samples
	while σᒾᵥ <= 0: σᒾᵥ = normal(σᒾₒ, 0.5) # keep generating until σᒾᵥ is a positive value.

	q = rand()
	if ln(q) < (p1(σᒾᵥ, w).item() - p1(σᒾₒ, w).item()): 
		σᒾₒ = σᒾᵥ 		# since we use ln(p(x)), the ratio is subtraction

	σᒾ_samples.append(σᒾₒ)
		   
	#	sampling w
	wᵥ = np.reshape(np.random.multivariate_normal(w.flatten(), Σᵥ, 1), (2,1))
	if ln(rand()) < (p2(σᒾₒ, wᵥ) - p2(σᒾₒ, w)): w = wᵥ
	w_samples = np.vstack((w_samples, w.flatten()))

σᒾ_samples = σᒾ_samples[burn:]
w_samples = w_samples[burn:, :]

print('E[μ] : ', np.mean(w_samples, axis=0))
print('E[σᒾ] : ', np.mean(σᒾ_samples))

import pdb; pdb.set_trace()
