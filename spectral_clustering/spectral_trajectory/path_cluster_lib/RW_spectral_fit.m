
function [centroid, pointsInCluster, assignment] = RW_spectral_fit(distance_matrix, cluster_N)

	%	Using Gaussian distance
	Adjacency_matrix = exp(-distance_matrix);
	Degree_matrix = diag(sum(Adjacency_matrix));

	Laplacian = inv(Degree_matrix)*Adjacency_matrix;
	[V,D] = eig(Laplacian);

%	figure(7);plot(diag(D));
%	%[s,id] = sort(diag(D));
%
	result = V(:,2:cluster_N);
	[centroid, pointsInCluster, assignment]= octave_kmeans(result, cluster_N);

end
