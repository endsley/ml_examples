#!/usr/bin/octave --silent

%	initialization
	cluster_N = 4;
	[y_normalized, y_total, N] = sample_data_generation(1,0);
	y_original = y_normalized;



	%	Calculate the distance
	distance_matrix_1 = get_Distance_in_Time(y_normalized);
	distance_matrix_2 = get_Distance_in_Freq(y_normalized, 0.2);

	p = 1.0;
	distance_matrix = p*distance_matrix_1 + (1-p)*distance_matrix_2;
	Adjacency_matrix = exp(-distance_matrix);
	assignment = zeros(size(y_normalized,2),1);
	
	[clusts,best_group_index,Quality,Vr] = cluster_rotate(Adjacency_matrix,[2:10],1,1);
	NN = size(clusts{best_group_index},2)
	for m = 1:NN
		p = clusts{best_group_index};
		assignment(p{1,m}) = m;
	end

	x = 1:lenght(y_original);
	plot_cluster_results(x, assignment, y_original, N, 10);

