
%	Data is assume to have each row as a single sample
function allocation = spectral_clustering(data, num_clusters, sigma)
	N = size(data,1);
	K = zeros(N,N);	
	for a = 1:N
		for b= 1:N
			K(a,b) = exp(-((data(a,:) - data(b,:))*(data(a,:) - data(b,:))')/(2*sigma));
		end
	end

	D = diag(1./sqrt(sum(K)));
	L = D*K*D;

	[eVec, eVal] = eig(L);
	[a,indx] = sort(diag(eVal),'descend');
	sVec = eVec(:, indx);
	U = sVec(:, 1:num_clusters);

	normalized_U = U./repmat(norm(U,2, 'rows'), 1, size(data,2));

	[centroid, pointsInCluster, assignment] =  octave_kmeans(normalized_U, num_clusters)
	
	allocation = 0
end
