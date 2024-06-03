#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, expon, gamma
from numpy.random import randn
from numpy import mean, linspace, sum
from scipy.special import gamma as Γ
from scipy.stats import norm
from scipy.integrate import quad 
from numpy import inf as ᨖ
from numpy import exp as e

#	# Parameter Estimation For Exponential Distribution
#	In this example, given data $\mathcal{D}$, we are going to estimate the parameters of an Exponential Distribution using
#	- Bayesian Parameter Estimation
#	- MAP
#	- Note that the equation for an exponential distribution is
#	$$ p(x) = \lambda e^{-\lambda x} $$
#	$$ E[X] = \frac{1}{\lambda}$$
#	- For Exponential distribution, the conjugate prior is the Gamma distribution where
#	$$ p(\lambda) = \frac{1}{\Gamma(k) \theta^k} \lambda^{k-1} e^{-\lambda/\theta}.$$


#	### The likelihood, Prior, and the Posterior
#	Given n samples, the likelihood function is
#	$$ p(X = \mathcal{D}|\lambda) = \prod_i \; \lambda e^{- \lambda x_i} = \lambda^n e^{-\lambda \sum_i \; x_i}$$.
#	Given the joint likelihood function and the conjugate prior, we know that $p(\lambda|X=\mathcal{D}) \propto  p(X=\mathcal{D}|\lambda)p(\lambda)$, therefore
#	$$p(\lambda|X=\mathcal{D}) \propto \left( \lambda^n e^{-\lambda \sum_i \; x_i} \right) \left( \frac{1}{\Gamma(k) \theta^k} \lambda^{k-1} e^{-\lambda/\theta} \right).$$
#	If we combine the terms together, we get
#	$$p(\lambda|X=\mathcal{D}) \propto \frac{1}{\Gamma(k) \theta^k} \lambda^{k+n-1} e^{-\lambda (\sum_i x_i + \frac{1}{\theta})}$$
#	Since $p(\lambda|X=\mathcal{D})$ is in terms of $\lambda$, the term $\frac{1}{\Gamma(k) \theta^k}$ at the front is just a constant, therefore
#	$$p(\lambda|X=\mathcal{D}) = \eta \lambda^{k+n-1} e^{-\lambda (\sum_i x_i + \frac{1}{\theta})}$$
#	- Pay special attention to the fact that we went from $\propto$ to = sign. 
#	- This is because we may not know the exact proportion of $\eta$, but we the posterior is equal to a constant multiple
#	- Once we simplified the equation, we notice that the structure of the residual equation is identical to a gamma distribution if
#	$$\hat{k} = k + n, \quad \quad \theta' = \frac{1}{\sum_i x_i - 1/\theta}$$
#	Giving us the posterior distribution
#	$$ p(\lambda|X=\mathcal{D}) = \frac{1}{\Gamma(\hat{k}) \; \theta'^\hat{k}} \; \lambda^{\hat{k}-1} e^{-\lambda/\theta'}.$$


#	#### Assume data is exponential, we will synthetically generate the data here
n = 1000
μ = 3
X = expon.rvs(scale=μ, size=n)


#	#### Define the prior gamma distribution as
k = 2
θ = 2
ǩ = k + n
θˊ = 1/(sum(X) + 1/θ)


#	#### Mode is the MAP solution
#	mode = $(k-1) \theta$
#	mean = $k \theta$
MAP_solution = (ǩ - 1)*θˊ
BPE_solution = ǩ*θˊ



print('Theoretical Truth = %.4f'%(1/μ))
print('Best λ according to MAP = %.4f'%MAP_solution)
print('Best λ according to MLE = %.4f'%(1/mean(X)))
print('Best λ according to Bayesian Parameter Estimation = %.4f'%(BPE_solution))


##	### Plotting out the Prior and posterior
##	In this example, we are going to assume a Gamma Prior with k = 2
x = linspace(0.2,0.5, 200);
y = gamma.pdf(x, ǩ, loc=0, scale=θˊ);

plt.figure(figsize=(13,4))
plt.subplot(131);
plt.hist(X, density=True, bins=30);
plt.title('Histogram of X')

plt.subplot(132)
plt.plot(x,y,color='red')
plt.title('Posterior Distribution of λ')

x = linspace(0,20, 200);
prior = gamma.pdf(x, k, loc=0, scale=θ);

plt.subplot(133)
plt.plot(x,prior,color='blue')
plt.title('Prior Distribution of λ')
plt.show()


#	### Notice
#	- The prior distribution says that the possible $\lambda$ values can possibly range between 0 to 20
#	- However, given the data, notice that the posterior distribution heavily indicates that $\lambda$ must be very close to 0.333.


#	## Finding the Prdictive Posterior $p(x|\mathcal{D})$ 
#	Our goal is to next find the $p(x|\mathcal{D})$ which has the equation
#	$$p(x|\mathcal{D}) = \int p(x|\lambda) p(\lambda|\mathcal{D}) \; d\lambda$$
#	Remember that in the previous step, we already found
#	$$ p(\lambda|X=\mathcal{D}) = \frac{1}{\Gamma(\hat{k}) \; \theta'^\hat{k}} \; \lambda^{\hat{k}-1} e^{-\lambda/\theta'}.$$
#	Therefore, the Predictive Posterior becomes
#	$$p(x|\mathcal{D}) = \int \; \left(\lambda e^{-\lambda x} \right) \left( \frac{1}{\Gamma(\hat{k}) \; \theta'^\hat{k}} \; \lambda^{\hat{k}-1} e^{-\lambda/\theta'} \right) \; d\lambda$$
#	Let's now multiply them together, and rearrange
#	$$p(x|\mathcal{D}) = \frac{1}{\Gamma(\hat{k})  \; \theta'^\hat{k}} \int \;   \; \lambda^{\hat{k}} e^{-\lambda (x + 1/\theta')} \; d\lambda$$
#	Here, notice that the remaining terms in the integral looks almost like another Gamma distribution. Let's try to make it look exactly like a gamma distribution by rearranging it
#	$$p(x|\mathcal{D}) = \frac{1}{\Gamma(\hat{k})  \; \theta'^{\hat{k}}} \int \;   \; \lambda^{(\hat{k}+1) - 1} e^{-\lambda (x + 1/\theta')} \; d\lambda$$
#	Now if we do a variable change, we can set it such that
#	$$k'' = \hat{k}+1, \quad \theta'' = \frac{1}{(x + 1/\theta')}$$
#	then
#	$$p(x|\mathcal{D}) = \frac{1}{\Gamma(\hat{k})  \; \theta'^{\hat{k}}} \int \;   \; \lambda^{k'' - 1} e^{-\lambda/\theta''} \; d\lambda.$$
#	It is very easy to now see that the terms inside the integral looks like a Gamma distribution, this implies that we could make it look exactly like Gamma by multiply it with 1 
#	$$p(x|\mathcal{D}) = \frac{\Gamma(k'')  \; \theta''^{k''}}{\Gamma(\hat{k})  \; \theta'^{\hat{k}}} 		\int \;   \frac{1}{\Gamma(k'')  \; \theta''^{k''}}  \; \lambda^{k'' - 1} e^{-\lambda/\theta''} \; d\lambda.$$
#	The reason why we put what's inside the integral to be exactly a Gamma distribution is because the integral of any distribution is always = 1, allowing us to remove the term.  
#	$$p(x|\mathcal{D}) = \frac{\Gamma(k'')  \; \theta''^{k''}}{\Gamma(\hat{k})  \; \theta'^{\hat{k}}} .$$

#	Remember that $\Gamma(n) = (n-1)!$, therefore
#	$$\frac{\Gamma(k'')}{\Gamma(\hat{k})} = \frac{(1003 - 1)!}{(1002 - 1)!} = 1002 = \hat{k}.$$
#	Also remember that
#	$$\theta' = \frac{1}{\sum_i x_i + 1/\theta} = \frac{\theta}{\theta \sum_i x_i + 1}.$$
#	and
#	$$\theta'' = \frac{1}{x + 1/\theta'} = \frac{1}{\theta x/\theta + (\theta \sum_i x_i + 1)/\theta} = \frac{\theta}{\theta x + \theta \sum_i x_i + 1}$$
#	therefore, 
#	$$\frac{\theta''^{k''}}{\theta'^{\hat{k}}} = \frac{\theta^{1003}}{(\theta x + \theta \sum_i x_i + 1)^{1003}} \frac{(\theta \sum_i x_i + 1)^{1002}}{\theta^{1002}} = \theta \left( \frac{\theta \sum_i x_i + 1}{\theta x +  \theta \sum_i x_i + 1} \right)^{1002} \frac{1}{\theta x + \theta \sum_i x_i + 1}$$
#	If we let $\alpha = \theta \sum_i x_i + 1$ then
#	$$p(x|\mathcal{D}) = \theta \hat{k} \left( \frac{\alpha}{\theta x + \alpha} \right)^{1002} \frac{1}{\theta x + \alpha}.$$


def posterior(x):
	α = θ*sum(X) + 1
	out = θ*ǩ*(α/(θ*x + α))**1002/(θ*x + α)
	return out


#	### Making sure the posterior is still a probability distribution
#	- when taking an integral from 0 to ᨖ , it should add up to 1 
I, err = quad(posterior, 0, ᨖ ) 
print('The area of the probability = %.4f, confirm that it is a proper probability distribution'%I) 


#	### Generate the predictive posterior distribution histogram via ancestral sampling
#	- Since we have the joint distribution, we can obtain the marginal by generating samples of $(x, \lambda)$ from the joint distribution and discard the $\lambda$ values.
#	- To generate the samples, we can use ancestral sampling. 
#	- That is, we first sample from the gamma distribution, depending on the $\lambda$ generated, we can generate $x$ using the exponential distribution.

posterior_samples = []
λs = gamma.rvs(ǩ, loc=0, scale=θˊ, size=n);
for λ in λs: posterior_samples.append(expon.rvs(loc=0, scale= (1/λ)))

#	### Point estimation using Monte Carlo Estimation
#	- We can use ancestral sampling to get the histogram of the posterior. 
#	- However, we can also get the point estimation of posterior at a specific point $p(x = \hat{x}|\mathcal{D})$ using monte carlo integration
#	$$p(x=3|\mathcal{D}) = \int p(x=3|\lambda) p(\lambda|\mathcal{D}) \; d\lambda \approx \frac{1}{n} \sum_i \; p(x=3|\lambda)$$
#	- Here is an example of $p(x = \hat{x}|\mathcal{D})$
p3 = mean(λs*e(-λs*3))
p3v = posterior(3)
print('Estimation of p(x=3) = %.4f, Actual p(x=3) = %.4f'%(p3, p3v))


#	### Let's plot out the predictive posterior 
#	- We will now plot out the posterior function as well as the histogram generated via sampling
x = linspace(0, 20, 100)
y = posterior(x)

plt.plot(x,y,color='red')
plt.hist(posterior_samples, density=True, bins=30);
plt.title('Predictive Posterior Distribution of x')
plt.show()


