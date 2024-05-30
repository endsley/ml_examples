#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from numpy import mean
from scipy.stats import norm

#	# Parameter Estimation For Exponential Distribution
#	In this example, given data $\mathcal{D}$, we are going to estimate the parameters of an Exponential Distribution using
#	- Bayesian Parameter Estimation
#	- MAP
#	- Note that the equation for an exponential distribution is
#	$$ p(x) = \lambda e^{-\lambda x} $$
#	$$ E[X] = \frac{1}{\lambda}$$
#	- For Exponential distribution, the conjugate prior is the Gamma distribution where
#	$$ p(\lambda) = \frac{1}{\Gamma(k) \theta^k} \lambda^{k-1} e^{-\lambda/\theta}.$$
#	### The likelihood function
#	Given n samples, the likelihood function is
#	$$ p(X = \mathcal{D}|\lambda) = \prod_i \; \lambda e^{- \lambda x_i} = \lambda^n e^{-\lambda \sum_i \; x_i}$$.
#	Given the joint likelihood function and the conjugate prior, we know that $p(\lambda|X=\mathcal{D}) \propto  p(X=\mathcal{D}|\lambda)p(\lambda)$, therefore
#	$$p(\lambda|X=\mathcal{D}) \propto \left( \lambda^n e^{-\lambda \sum_i \; x_i} \right) \left( \frac{1}{\Gamma(k) \theta^k} \lambda^{k-1} e^{-\lambda/\theta} \right).$$
#	If we combine the terms together, we get
#	$$p(\lambda|X=\mathcal{D}) \propto \frac{1}{\Gamma(k) \theta^k} \lambda^{k+n-1} e^{-\lambda (\sum_i x_i - \frac{1}{\theta})}$$
#	Since $p(\lambda|X=\mathcal{D})$ is in terms of $\lambda$, the term $\frac{1}{\Gamma(k) \theta^k}$ at the front is just a constant, therefore
#	$$p(\lambda|X=\mathcal{D}) = \eta \lambda^{k+n-1} e^{-\lambda (\sum_i x_i - \frac{1}{\theta})}$$
#	- Pay special attention to the fact that we went from $\propto$ to = sign. 
#	- This is because we may not know the exact proportion of $\eta$, but we the posterior is equal to a constant multiple
#	- Once we simplified the equation, we notice that the structure of the residual equation is identical to a gamma distribution if
#	$$\hat{k} = k + n, \quad \quad \hat{\theta} = \frac{1}{\sum_i x_i - 1/\theta}$$
#	Giving us the posterior distribution
#	$$ p(\lambda|X=\theta) = \frac{1}{\Gamma(\hat{k}) \; \hat{\theta}^\hat{k}} \; \lambda^{\hat{k}-1} e^{-\lambda/\hat{\theta}}.$$

