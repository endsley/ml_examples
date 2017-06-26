%	Input argument
% A : is the data, where each sample is a single column
% EV_percentage : this controls what percentage of emphasis 1.00 is completely time domain and 0 is completely Freq domain
% remove_percentage : percentage of data we remove for variance map, 1 is 100%
% plot_it : 1 to display plot and 0, not to
%
%	Output Argument
% assignment : the assignment of the clustering
% NN : the number of determined clusters
function [assignment, NN ]= spectral_path_clustering(A, EV_percentage, remove_percentage)
	model_order = 50;
	use_tigher_bound = 0;


	if(use_tigher_bound == 1)
		%	Calculate the distance
		get_Distance_in_Time(A, EV_percentage, remove_percentage, use_tigher_bound);
		load similarity_matrix.mat;
		similarity_matrix_1 = similarity_matrix;
	
		get_Distance_in_Freq(A, (1-EV_percentage), remove_percentage, use_tigher_bound);
		load similarity_matrix.mat;
		similarity_matrix_2 = similarity_matrix;
	
		%load similarity_matrix.mat;
		similarity_matrix = EV_percentage*similarity_matrix_1 + (1-EV_percentage)*similarity_matrix_2;
		Adjacency_matrix = similarity_matrix .* similarity_matrix';
		Adjacency_matrix = Adjacency_matrix - diag(diag(Adjacency_matrix));
	else
		%	Calculate the distance
		distance_matrix_1 = get_Distance_in_Time(A, EV_percentage, remove_percentage, use_tigher_bound);
		distance_matrix_2 = get_Distance_in_Freq(A, (1-EV_percentage), remove_percentage, use_tigher_bound);
	
		distance_matrix = EV_percentage*distance_matrix_1 + (1-EV_percentage)*distance_matrix_2;
		Adjacency_matrix = exp(-distance_matrix);
		Adjacency_matrix = Adjacency_matrix - diag(diag(Adjacency_matrix));
	end

	if(length(Adjacency_matrix) < 10)
		model_order = length(Adjacency_matrix);
	end
	assignment = zeros(size(A,2),1);
	
	[clusts,best_group_index,Quality,Vr] = cluster_rotate(Adjacency_matrix,[2:model_order],1,1);
	NN = size(clusts{best_group_index},2);
	for m = 1:NN
		p = clusts{best_group_index};
		assignment(p{1,m}) = m;
	end
end
