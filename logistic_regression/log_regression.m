#!/usr/bin/octave --silent

%	 This is an example of using octave logistic regression function
%	 Let x be the input and y as the output, each column is a single sample
x = [0 1 2 3 4 4 5 5 9 9 10 11 12 13 ]';
y = [0 0 0 0 0 1 0 1 1 1 1 1 1 1 ]';
%y = [0 0 0 0 0 0 0 1 1 1 1 1 1 1 ]';


[theta, beta] = logistic_regression(y,x)

x2 = [0:0.2:13]';
v = theta - beta*x2;
out = exp(-v)./(1+exp(-v))
plot(x2, out)
input('')







%	Here is my own implementation of 

%w = [1;1];
%x = [ones(1,length(x));x];
%alpha = 0.10;
%
%for m = 1:100
%	a = (1./(1+exp(w'*x)));
%	gradient = x*a' - x*y';
%	w = w + alpha*gradient
%
%	%cost = -(log(a)*y' + log(1-a)' - log(1-a)*y')
%end
%
%
%x2 = -0:0.2:13;
%x3 = [ones(1,length(x2));x2];
%
%a = (1./(1+exp(w'*x3)));
%%plot(x2, a, 'b')
%plot(x2, a)
%input('')
%%x*(1./(1+exp(w*x)))' - x*y'
