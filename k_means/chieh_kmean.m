% Input
%	r : the number of kmean iteration, to avoid bad seed
%	x : the observed data, each column is each observation
%	k : the number of clusters
% Output
%	cluster : 

function clusters = chieh_kmean(r, x, k)
	clib = cluster_lib(x,k);
	criteria_met = 0;
	clusters = containers.Map();
	clusters('square_sum_error') = realmax('double');
	clusters('mean_matrix') = [];
	clusters('classification') = [];
	clusters('error_convergence') = [];

	for count = 1:r
		%	Initialize seed
		seed_mean = clib.initialize_seed();
		previous_mean = seed_mean;
		error_convg = [];

		while(criteria_met == 0)
			%	Classify
			[distance_matrix, square_sum_error] = clib.calculate_distance_matrix(previous_mean);
			classified = clib.hard_classification(distance_matrix);
		
			%	Recalculate mean
			next_mean = clib.mean_recalculation(classified);
		
			%	Criteria met?
			[criteria_met, previous_mean, L_infinity] = clib.k_mean_criteria_met(previous_mean, next_mean);
			error_convg = [error_convg, L_infinity];
		end
		criteria_met = 0;

		[error_matrix, square_sum_error] = clib.calculate_distance_matrix(previous_mean);
		square_sum_of_errors = sum(sum(error_matrix.*error_matrix));

		if(square_sum_of_errors < clusters('square_sum_error'))
			clusters('square_sum_error') = square_sum_of_errors;
			clusters('seed_mean') = seed_mean;
			clusters('mean_matrix') = previous_mean;
			clusters('classification') = classified;
			clusters('error_convergence') = error_convg;
		end
	end

	clib.plot_result(clusters, 'Geyser data', 2);
end
