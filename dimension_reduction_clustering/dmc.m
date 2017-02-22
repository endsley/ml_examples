#!/usr/bin/octave --silent

X = [0.2*randn(10,4); 0.2*randn(10,2)+5, 0.2*randn(10,2)];
n = size(X)(1);
d = size(X)(2);
A = eye(d);
H = eye(n) - (1.0/n)*ones(n,n);
converged = 0;
k = 2;
delta = 0.001;
lambda = 0.5;


%while(converged == 0)

C = H*X*A*X'*H;
[V,D] = eig(C);
U = V(:,1:k);
[centroid, pointsInCluster, assignment] =  octave_kmeans(U, k);
assignment

for n = 1:3

	for m = 1:4
		part_1 = inv(A + delta*eye(d));
		part_2 = X'*H*U*U'*H*X;
	
		n_1 = trace(part_1);
		n_2 = trace(part_2);
		%lambda = n_1/n_2;
	
		FI = part_1 - lambda*part_2;
		[V, D] = sorted_eig(FI, 'ascend');
	
		below_0_eigs = diag(D)(diag(D) < 0);
		dimension_to_keep = length(below_0_eigs);
		L = V(:,1:dimension_to_keep);
		%L = V(:,1:2);
		if( norm(L*L' - A, 'fro') < 0.001)
			'broke'
			break;
		end
		A = L*L';
	end

	rA = rank(A);
	C = H*X*A*X'*H;
	[V,D] = eig(C);

end

U = U(:,rA)

[centroid, pointsInCluster, assignment] =  octave_kmeans(U, k);
assignment
plot(U,'x')

