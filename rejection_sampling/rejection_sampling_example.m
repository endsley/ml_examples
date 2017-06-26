#!/usr/bin/octave

%	In this example, we want to generate samples from Gamma distribution
%						1      α   α -1    -βz
%		Gam(z|α,β) =  -----  β    z      e 				Let α = 2 , β  = 2
%					   Γ(α)
%
%	With rejection sampling, we can just ignnore the constants and use 
%
%					    -2z
%		Gam(z|α,β) = z e 
%					
%


x = 0:0.1:8;
p = x.*exp(-2*x);
sample_size = 10000;
proposal =  0.25*exp(-0.25*x);					% exponential distribution function
expR = log(1 - rand(1,sample_size))/(-0.25);	% Generate samples

%--------    accept only if 
cutoff = expR.*exp(-2*expR);					% get y1 value at each sample for gamma
exp_loc = (0.25*exp(-0.25*expR));				% get y2 value at each sample for the proposal distribution
uniform_guess =  rand(1,sample_size).*exp_loc;	% get uniform from 0 - y2
gamma_samples = expR(uniform_guess < cutoff);

plot(x,p); hold on;
plot(x,proposal); 
%hist(expR,33,1)
hist(gamma_samples,22,1)

axis([0,15,0,0.3]);
input('');
