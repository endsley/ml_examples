
function [centroid, pointsInCluster, assignment] = spectral_fit(Adjacency_matrix, cluster_N)
	%Adjacency_matrix = [1 1 0 0 0;1 1 0 0 0;0 0 1 1 0; 0 0 1 1 1;0 0 0 1 1]

	%------------------------------

	Degree_matrix = diag(sum(Adjacency_matrix));
	
	% Shi and Malik Method
	%	Find max inv(D)*W 
	Laplacian = Degree_matrix - Adjacency_matrix;
	[V,D] = eig(inv(Degree_matrix)*Adjacency_matrix);

	[a,indx] = sort(diag(D),'descend');
	sorted_eigenVectors = V(:, indx);
	result = sorted_eigenVectors(:,2:cluster_N);
	
	[centroid, pointsInCluster, assignment]= octave_kmeans(result, cluster_N);

end

%	Direct svd approach	

%	[V,D] = eig(Adjacency_matrix);
%	[a,indx] = sort(diag(D),'descend');
%	sorted_eigenVectors = V(:, indx);
%	result = sorted_eigenVectors(:,2:cluster_N);
%	[centroid, pointsInCluster, assignment]= octave_kmeans(result, cluster_N);



