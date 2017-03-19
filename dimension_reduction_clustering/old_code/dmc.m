%#!/usr/bin/octave --silent

noise_level = 1;
n = 20;
dim_feature = 2;
dim_noise = 2;
d = dim_feature + dim_noise;
X = [0.2*randn(n/2, dim_feature), noise_level*rand(n/2,dim_noise); 0.2*randn(n/2,dim_feature)+5, noise_level*rand(n/2,dim_noise)];
A = eye(d);
H = eye(n) - (1.0/n)*ones(n,n);
converged = 0;
k = 2;
delta = 0.001;



%[centroid, pointsInCluster, assignment] =  octave_kmeans(X, k);
%assignment


%while(converged == 0)

C = H*X*A*X'*H;
[V,D] = eig(C);
U = V(:,1:k);
[centroid, pointsInCluster, assignment] =  octave_kmeans(U, k);
lambda = 1;
%assignment

for n = 1:3

	for m = 1:4
		part_1 = inv(A + delta*eye(d));
		part_2 = X'*H*U*U'*H*X;
	
		n_1 = norm(part_1,'fro');
		n_2 = norm(part_2,'fro');
		%lambda = n_1/n_2;
	
		FI = part_1 - lambda*part_2;
		[V, D] = sorted_eig(FI, 'ascend');

		tmp_D = diag(D)
		below_0_eigs = tmp_D(diag(D) < 0);
		dimension_to_keep = length(below_0_eigs);
		L = V(:,1:dimension_to_keep);
		%L = V(:,1:2);
		if( norm(L*L' - A, 'fro') < 0.001)
			%'broke'
			break;
		end
		A = L*L';
	end

	rA = rank(A);
	C = H*X*A*X'*H;

	[U_new, S, V] = svd(C);
	U = U_new(:,rA)

	dbstop
end


[centroid, pointsInCluster, assignment] =  octave_kmeans(U, k);
assignment
%rA
%plot(U,'x')

