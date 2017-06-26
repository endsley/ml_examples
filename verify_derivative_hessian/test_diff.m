#!/usr/bin/octave
% This script numerically calculate the gradient and the hessian
% to verify if a gradient or hessian formula is correct

%	Example : Given  f(x) = 1/2*x'Qx - bx + c
%					df(x) = Qx - b
%				  d^2f(x) = Q

%% generate data
l = 3;
x = rand(l,1);
Q = randn(l,l);
b = rand(l,1);
c = 5;

addpath(genpath('DERIVESTsuite'));

%% get derivative/Hessian numerically,  gn = gradient numerical
%% put the f(x) into compute_J.m
F_x = @(x)compute_J(x, Q,b,c);
gn = gradest(F_x,x);
numerical_gradient = reshape(gn, l, 1);
numerical_Hessian = hessian(F_x,x);

% get derivative/Hessian using equation
theoretical_gradient = 0.5*(Q+Q')*x - b;
theoretical_Hessian = 0.5*(Q+Q');

%%% get the differential 
gradient_error = norm(numerical_gradient - theoretical_gradient, 'fro')/norm(theoretical_gradient, 'fro');
Hessian_error = norm(numerical_Hessian - theoretical_Hessian, 'fro')/norm(theoretical_Hessian, 'fro');
printf(['Gradient error : ' num2str(gradient_error)  '\n']);
printf(['Hessian error : ' num2str(Hessian_error)  '\n']);
