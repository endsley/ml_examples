#!/usr/bin/octave --silent

%	initialization
	cluster_N = 4;
	[y_normalized, y_total, N] = sample_data_generation(1);

	fft_filtered = conv_to_freq(y_normalized, 0, 0.9);
	%size(fft_filtered)
	%stem(fft_filtered(:,1))

	time_filter_dat = variance_map_filter(y_normalized);
	freq_filter_dat = variance_map_filter(fft_filtered);


%	Calculate the sigma value
	L2_distance = calc_Eucli_Distance_matrix(time_filter_dat, 0);
	sigma = median(L2_distance(:));

%	sigma_max = 2*max(L2_distance(:))^2
%	sigma_min = 2*min(L2_distance(:))^2

%	Calculate Distance Matrix
	distance_matrix_1 = calc_Jensen_shannon_divergence(freq_filter_dat);
	distance_matrix_2 = calc_Eucli_Distance_matrix(freq_filter_dat, 1);
	distance_matrix_3 = calc_Jensen_shannon_divergence(time_filter_dat);
	distance_matrix_4 = calc_Eucli_Distance_matrix(time_filter_dat);




	p = 0.7;
	distance_matrix = p*(distance_matrix_1 + distance_matrix_2) + (1-p)*(distance_matrix_3 + distance_matrix_4);

%%dist_3 = max(max(distance_matrix_3))
%%dist_4 = max(max(distance_matrix_4))
%
%
%%	distance_matrix = distance_matrix_1;
%
%
%	fit to spectral clustering
	[centroid, pointsInCluster, assignment] = spectral_fit(distance_matrix, cluster_N, sigma);

	plot_cluster_results(assignment, y_normalized, N);
input('')
