#!/usr/bin/octave


k = 3; 
n = 2*k;

%A = [1 8 3 2;9 3 8 2;1 9 3 0;1 1 3 4];
A = 30*floor(randn(10,10));

A_width = size(A,2);
[omg, R] = qr(randn(A_width, n));

omg = omg(:,1:n);

%[Q,R] = qr(A*omg,'0');
[Q,R] = qr((A*A')^3*A*omg,'0');

smaller_matrix = Q'*A;
size(smaller_matrix)
[U,S,V] = svd(smaller_matrix);
new_S = diag(S);
new_S = sort(new_S,'descend');
Sa = new_S(1:k)
estimated_U = Q*U;
estimated_U = estimated_U(:,1:k)


[U,S,V] = svd(A);
new_S = diag(S);
new_S = sort(new_S,'descend');
Sb = new_S(1:k)
real_U = U(:,1:k)

(Sa - Sb)./Sb

