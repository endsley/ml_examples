
function cluster_matrix = form_cluster_matrix(labels)
	labels = labels(:);

	cluster_matrix = repmat(labels, 1, length(labels)) - repmat(labels',length(labels), 1);
	cluster_matrix = cluster_matrix == 0;
end
