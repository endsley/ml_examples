
function [centroid, pointsInCluster, assignment] = spectral_fit(distance_matrix, cluster_N, sigma)
	sigma = std(distance_matrix(:));
	sss = sigma^2*2

%	%	Using information
%	distance_matrix = distance_matrix + 0.01*eye(size(distance_matrix));
%	Adjacency_matrix = log(distance_matrix./sum(sum(distance_matrix)));

	%	Using Gaussian distance
	pppppppppppppppp = 2*sigma^2
	Adjacency_matrix = exp(-distance_matrix/(2*sigma^2));
	%Adjacency_matrix = exp(-distance_matrix/(0.1));  % 0.1 seems to be the best


	%[V2, D2] = eig(Adjacency_matrix)
	%Adjacency_matrix = -2*distance_matrix + 2*max(max(distance_matrix));


	Degree_matrix = diag(sum(Adjacency_matrix));
	Laplacian = Degree_matrix - Adjacency_matrix;
	[V,D] = eig(Laplacian);
	diag(D)
	figure(7);plot(diag(D));
	%[s,id] = sort(diag(D));

	result = V(:,2:cluster_N);
	[centroid, pointsInCluster, assignment]= octave_kmeans(result, cluster_N);

end
