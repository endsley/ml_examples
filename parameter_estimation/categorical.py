#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from numpy import mean
from scipy.stats import norm

#	# Parameter Estimation for Categorical Distribution
#	In this example, we are going to estimate the parameters of a Categorical Distribution with 3 categories
# 	where the data $\mathcal{D}$ consists of 1 red, 2 blue, and 4 green.

#	## Basic counting and recognize distribution
#	Using this approach, we simply need to recognize that this situation has 3 outcomes.
#	The Categorical Distribution is commonly used as the default structure of the pdf.
#	In general, if we approximate p(x) with Categorical, then we assume a structure of 
#	$$p(x) = \theta_1^{x_1} \theta_2^{x_2} \theta_3^{x_3} \quad \text{where} \quad x_1,x_2,x_3 \in \{0,1\} \; and \; x_1 + x_2 + x_3 = 1.$$

#	According to MLE, $\theta_i$ is the probability of success for each category.
#	Given 1 red, 2 blue, and 4 green, the p(x) would be:

#	$$p(x) = (1/7)^{x_1} (2/7)^{x_2} (4/7)^{x_3}$$

#	Note that $x_1, x_2, x_3$ here denotes the category and __not the sample id__. 

#	## Maximum A Posteriori Estimation (MAP)
#	The MAP estimation is more complicated. Here, instead of maximizing the $p(X=\mathcal{D}|\theta)$, we want to find $\max \; p(\theta|X=\mathcal{D})$. 
#	In other word, we want to find the most likely $\theta$ giving the entire dataset $X=\mathcal{D}$.
#	Take a quick second to distinguish the difference between MLE and MAP
#	- MLE : 
#	$$ \max_{\theta} \; p(X=\mathcal{D}|\theta) $$
#	- MAP : 
#	$$ \max_{\theta} \; p(\theta|X=\mathcal{D}) $$
#	With this method, we use the Bayes' Theorem 
#	$$ p(\theta | X=\mathcal{D}) = \frac{p(X=\mathcal{D}|\theta) p(\theta)}{p(X=\mathcal{D})} $$
#	From MLE, we knew tht 
#	$$p(X=\mathcal{D}|\theta) = \mathcal{L} = \prod_{i=1}^n \; \theta_1^{x_{i1}} \theta_2^{x_{i2}} \theta_3^{x_{i3}}$$
#	Note that for $x_{ij}$, $i$ represents the sample id, and $j$ represents the category id. 


#	- With MLE, the likelihood function is sufficient. 
#	- With MAP, it allow us to use prior knowledge about the distribution of $\theta$. The MAP estimate consequently combines our prior knowledge with the data and come up with the best estimation. 
#	- In this particular example, we use a Dirichlet distribution with $\alpha_1 = 2, \alpha_2 = 2, \alpha_3 = 2$. 
#	$$ p(\theta) = \frac{\Gamma(\alpha_1 + \alpha_2 + \alpha_3)}{\Gamma(\alpha_1)\Gamma(\alpha_2)\Gamma(\alpha_3)} \theta_1^{\alpha_1 - 1} \theta_2^{\alpha_2 - 1} \theta_3^{\alpha_3 - 1} \quad \text{where if $n$ is an integer then} \quad \Gamma(n) = (n-1)!$$
#	Note that the $\Gamma(z)$ function is much more complicated if $z$ is not an integer, specifically, it is 
#	$$ \Gamma(z) = \int_0^{\infty} \; t^{z - 1} e^{-t} \; dt $$


#	###	Applying the Conjugate Priors
#	To obtain the posterior, we apply the conjugate prior of categorical distribution, which is a Dirichlet distribution
#	$$ p(\theta) = \frac{\Gamma(\alpha_1 + \alpha_2 + \alpha_3)}{\Gamma(\alpha_1)\Gamma(\alpha_2)\Gamma(\alpha_3)} \theta_1^{\alpha_1 - 1} \theta_2^{\alpha_2 - 1} \theta_3^{\alpha_3 - 1} = B(\alpha) \theta_1^{\alpha_1 - 1} \theta_2^{\alpha_2 - 1} \theta_3^{\alpha_3 - 1} \quad \text{where if $n$ is an integer then} \quad \Gamma(n) = (n-1)!$$
#	Therefore, the posterior is 
#	$$p(\theta|X=\mathcal{D}) = \frac{p(X=\mathcal{D}|\theta) p(\theta)}{p(X=\mathcal{D})}$$
#	implying that
#	$$p(\theta|X=\mathcal{D}) \propto \left( \prod_{i=1}^n \; \theta_1^{x_{i1}} \theta_2^{x_{i2}} \theta_3^{x_{i3}} \right) \left( B(\alpha) \theta_1^{\alpha_1 - 1} \theta_2^{\alpha_2 - 1} \theta_3^{\alpha_3 - 1}  \right)$$
#	$$p(\theta|X=\mathcal{D}) \propto \left( \theta_1^{\sum_i x_{i1}} \theta_2^{\sum_i x_{i2}} \theta_3^{\sum_i x_{i3}} \right) \left( B(\alpha) \theta_1^{\alpha_1 - 1} \theta_2^{\alpha_2 - 1} \theta_3^{\alpha_3 - 1}  \right)$$
#	$$p(\theta|X=\mathcal{D}) \propto B(\alpha) \left( \theta_1^{\sum_i x_{i1} + \alpha_1 - 1} \theta_2^{\sum_i x_{i2} + \alpha_2 - 1} \theta_3^{\sum_i x_{i3} + \alpha_3 - 1} \right).$$
#	This tells us that the posterior is also a Dirichlet distribution where we let
#	$$ \hat{\alpha_1} = \sum_i x{i1} + \alpha_1$$
#	$$ \hat{\alpha_2} = \sum_i x{i2} + \alpha_2$$
#	$$ \hat{\alpha_3} = \sum_i x{i3} + \alpha_3$$
#	Then
#	$$p(\theta|X=\mathcal{D}) = \frac{\Gamma(\hat{\alpha_1} + \hat{\alpha_2} + \hat{\alpha_3} )}{\Gamma(\hat{\alpha_1})\Gamma(\hat{\alpha_2})\Gamma(\hat{\alpha_3})} \left( \theta_1^{\hat{\alpha_1} - 1} \theta_2^{\hat{\alpha_2} - 1} \theta_3^{\hat{\alpha_3} - 1} \right).$$

