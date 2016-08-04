% This script tests the differentiation

addpath(genpath('DERIVESTsuite'));

%	Define some common constants 
x1 = [1;0;1]; x2 = [3;2;1];
d = 3; q = 2;
Z = randn(d,q); 
W = randn(d,q); 
L1 = randn(q,q);
L2 = randn(q,d);
	
%	Get function handler for the original function
%	@(W) says that W is the variable, and W,Z is the location of differentiation
F_W = @(W)compute_J(W,Z);
%F_W = @(W)compute_J(W,Z,L1,L2,q);
%F_W = @(W)compute_J(W,Z,x1,x2);


%	Compute our hand calculated version of derivative at W and Z
formula_derivative = 2*(W-Z);
%formula_derivative = L2';
%formula_derivative = Z*L1;
%formula_derivative = (-2*x1*x1'*W)*exp(-x1'*W*W'*x1) + 2*(-2*x2*x2'*W)*exp(-x2'*W*W'*x2)


%	Compute the numerical version of derivative at W and Z
numerical_derivative = gradest(F_W,W);


%	Calculate the difference between the 2 versions
derivative_error = sum(formula_derivative(:) - numerical_derivative(:))
example_view = [formula_derivative(:) numerical_derivative(:)];
if size(example_view,1) > 20
	example_view(1:20,:)
else
	example_view
end

