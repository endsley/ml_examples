#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from numpy import mean
from scipy.stats import norm

#	# Parameter Estimation
#	In this example, we are going to estimate the parameters of an Exponential Distribution using
#	- Bayesian Parameter Estimation
#	- MAP
#	- Note that the equation for an exponential distribution is
#	$$ p(x) = \lambda e^{-\lambda x} $$
#	$$ E[X] = \frac{1}{\lambda}$$
#	- For Exponential distribution, the conjugate prior is the Gamma distribution where
#	$$ p(\lambda) = \frac{1}{\Gamma(k) \theta^k} \lambda^{k-1} e^{-\lambda/\theta}.
