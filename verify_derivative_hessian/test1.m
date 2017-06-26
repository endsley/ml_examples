#!/usr/bin/octave
% This script numerically calculate the gradient and the hessian
% to verify if a gradient or hessian formula is correct

%	Example : Given  f(x) = -c * exp((-w'*A*w)/(2*sigma^2))
%					df(x) =  c/sigma^2 * exp((-w'*A*w)/(2*sigma^2)) * A w
%				  d^2f(x) = Q

%% generate data
d = 10;
w = [-0.577;0.577;0.577];
%w = [0.1;0.3;0.5];
%w = [1;1;1];
A1 =[1,0,1; 0,0,0;1,0,1];
A2 =[9,6,3;6,4,2;3,2,1];
%A2 =[1 1 2;1 1 2;2 2 4];

addpath(genpath('DERIVESTsuite'));

%% get derivative/Hessian numerically,  gn = gradient numerical
%% put the f(w) into compute_J.m
F_x = @(w)compute_J(w,A1, A2);
numerical_gradient = gradest(F_x,w);
numerical_Hessian = hessian(F_x,w)

%% get derivative/Hessian using equation
%theoretical_gradient = 4*exp(-w'*A2*w)*A2*w;
%theoretical_Hessian = 2*exp(-w'*A1*w)*(A1 - 2*A1*w*w'*A1') + 4*exp(-w'*A2*w)*(A2 - 2*A2*w*w'*A2')
%theoretical_Hessian = 2*exp(-w'*A1*w)*(A1 - 2*A1*w*w'*A1')
theoretical_Hessian = 4*exp(-w'*A2*w)*(A2 - 2*A2*w*w'*A2')
%theoretical_Hessian = (A2 - 2*A2*w*w'*A2')

%%%% get the differential 
%gradient_error = norm(numerical_gradient - theoretical_gradient, 'fro')/norm(theoretical_gradient, 'fro');
%Hessian_error = norm(numerical_Hessian - theoretical_Hessian, 'fro')/norm(theoretical_Hessian, 'fro');
%printf(['Gradient error : ' num2str(gradient_error)  '\n']);
%printf(['Hessian error : ' num2str(Hessian_error)  '\n']);
