#!/usr/bin/octave --silent




addpath('./path_cluster_lib')

%	initialization
	%cluster_N = 4;
	sample_data_id = 8;
	[y_normalized, y_total, N] = sample_data_generation(sample_data_id, 0);


	original{1} = y_normalized;
	%y_normalized = fft_filter(y_normalized, 0);
	data{1} = y_normalized;


%	clustering portion
	[assignment, NN ] = spectral_path_clustering(data, 1.0, 0.0);
	save(['./data/assignments_' num2str(sample_data_id) '.mat'],'assignment', 'sample_data_id' );

%	plotting
	x = [1:size(y_normalized,1)]';

	%plot_cluster_results(x, assignment, data{1}, N, 10);
	plot_cluster_results(x, assignment, original{1}, N, 11);
	%plot_2d_data(assignment, original{1}, N, 10);

	input('');

