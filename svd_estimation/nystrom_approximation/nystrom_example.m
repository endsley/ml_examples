#!/usr/bin/octave --silent

format compact

%	Generating Data, 4 by 4 matrix with rank of 2
n = 5000;
sampling_percentage = 0.4;
num_of_eig = 5;
seedM = floor(10*rand(n,n));
[Q,R] = qr(seedM);
V = Q(:,1:num_of_eig);
noise = diag(0.001*randn(1,n));
X = V*diag(1:num_of_eig)*V' + noise;


%	Perform sampling

l = floor(n*sampling_percentage);
rp = randperm(n);

W = X(rp(1:l),rp(1:l));
G21 = X(rp(l+1:end), rp(1:l));
G22 = X(rp(l+1:end),rp(l+1:end));
C = [W;G21];
%X_2 = [W G21';G21 G22];	<= This is the estimated matrix

%--------------------------------



r = n/(l);
[V,D] = eig(W);
E = r*sort(diag(D),'descend');
E(1:num_of_eig)





