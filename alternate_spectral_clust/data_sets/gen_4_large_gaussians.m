#!/usr/bin/octave

d = 2;
n = 100;
p = 10;

a  =  randn(n,d)  +  repmat([0,p],n,1);
b  =  randn(n,d)  +  repmat([p,0],n,1);
c  =  randn(n,d)  +  repmat([0,-p],n,1);
d  =  randn(n,d)  +  repmat([-p,0],n,1);

dat  =  [a;b;c;d];
noise = rand(400,4);

X = [dat, noise];
X = X - repmat(mean(X),  n*4,  1);
X = round(X*100)/100


plot(X(:,1), X(:,2), 'x');

%save KDAC_test.mat X
csvwrite('Four_dimond_gaussians.csv',X, 'precision', 3)
