
classdef cluster_lib < handle
	properties
		x;
		k;
		n;
		d;
		escape_criteria;
		avoid_singular;
		minimum_counter;
	end

	methods

		%	x : each data set is a column, with n elements each column
		%	k : the number of initial seed
		function obj = cluster_lib(input_x, input_k)
			obj.x = input_x;
			obj.k = input_k;
			obj.n = size(input_x,2);
			obj.d = size(input_x,1);
			obj.escape_criteria = 0.0000001;

			obj.avoid_singular = 0.00001*min(var(input_x'));
			obj.minimum_counter = 0;
		end

		function [c_mean, c_sigma, prior] = initialize_seed(obj)
			c_mean = obj.x(:, randperm(obj.n, obj.k));
			c_sigma = repmat(eye(obj.d),1,obj.k);
			prior = ones(obj.k,1)./(obj.k);
			obj.minimum_counter = 0;
		end
		
		function [distance_matrix, square_sum_error] = calculate_distance_matrix(obj, current_mean)
			x_large = repmat(obj.x,1, obj.k);
			mean_large = reshape(repmat(current_mean, obj.n,1),size(x_large));
			zero_meaned = x_large - mean_large;
			square_sum_error = sum(zero_meaned.*zero_meaned,1);
			L2 = sqrt(square_sum_error);
			distance_matrix = reshape(L2, obj.n, obj.k);
			square_sum_error = sum(min(reshape(square_sum_error, obj.n, obj.k)'));
		end

		function [prob_matrix, likelihood] = soft_classification(obj, c_mean, c_sigma, prior)
			all_points = repmat(obj.x', obj.k, 1);
			mean_matrix = reshape(repmat(c_mean, obj.n, 1), obj.d, obj.n*obj.k)';

			cov_matrix_3d = [];
			for m = 0:(obj.k-1)
				cov_rep = repmat(c_sigma(: , 1+m*obj.d:m*obj.d+obj.d), [1, 1, obj.n]);
				cov_matrix_3d = cat(3, cov_matrix_3d, cov_rep);
            end

            %mean_matrix
            %all_points
			guass_matrix = mvnpdf(all_points, mean_matrix, cov_matrix_3d);
			guass_matrix = reshape(guass_matrix, obj.n, obj.k);
			likelihood = guass_matrix.*repmat(prior', obj.n, 1);
			if(min(sum(likelihood,2)) == 0)
				likelihood = likelihood + 0.000001;
			end
			prob_matrix = likelihood./repmat(sum(likelihood,2), 1, obj.k);

		end

		function classified = hard_classification(obj, distance_matrix)
			[a, classified] = min(distance_matrix,[], 2);
		end

		function [c_mean, c_sigma, prior] = gmm_parameter_recalculation(obj, prob_matrix, prior, c_sigma)
			prior = sum(prob_matrix',2)./obj.n;
			c_mean = ((obj.x*prob_matrix)./(obj.n*(repmat(prior, 1, obj.d))'));

			c_sigma = [];

			%	Sigma
			x_large = repmat(obj.x,1, obj.k);
			mean_large = reshape(repmat(c_mean, obj.n,1),size(x_large));
			dist = x_large - mean_large;
		
			likelihood = prob_matrix(:);
			for i=0:(obj.k-1)
				cov_m = zeros(obj.d, obj.d);
				for j = 1:obj.n
					y_i = dist(:,j+(obj.n*i));
					c_i = likelihood(j+(obj.n*i));
					cov_m = cov_m + c_i*(y_i*y_i');
				end
		
				if(prior(i+1) == 0)
					prior(i+1) = 0.000001;
				end
				cov_m = cov_m./(obj.n*prior(i+1));
				if(min(eig(cov_m)) < 0.000001)
					cov_m = cov_m + obj.avoid_singular*eye(obj.d);
                end

				c_sigma = [c_sigma cov_m];
			end
		end

		function next_mean = mean_recalculation(obj, classified)
			next_mean = [];
			for m = 1:obj.k
				next_mean = [next_mean mean(obj.x(:,find(classified == m)),2)];
			end
		end

		function [criteria_met, log_likelihood] = gmm_criteria_met(obj, likelihood, previous_log_likelihod)
			log_likelihood = sum(log(sum(likelihood,2)));
			differential_likelihood = abs(log_likelihood - previous_log_likelihod);

			if(differential_likelihood < obj.escape_criteria)
				if(obj.minimum_counter == 5)	% prevent temporary slow convergence
					criteria_met = 1;
				else
					criteria_met = 0;
					obj.minimum_counter = obj.minimum_counter + 1;
				end
			else
				criteria_met = 0;
			end
		end

		function [criteria_met, previous_mean, L_infinity] = k_mean_criteria_met(obj, previous_mean, next_mean)
			%L infinity is faster, and the result is good enough
			%We can also use the Q function, or error distance, but L infinity is faster
			L_infinity = norm(previous_mean - next_mean, inf); 
			if(L_infinity < obj.escape_criteria)
				criteria_met = 1;
			else
				criteria_met = 0;
			end

			previous_mean = next_mean;
		end

		function print_outcome(obj, map_obj, text_title, text_key)
			if map_obj.isKey(text_key)
				fprintf([text_title ' : ' mat2str(round(map_obj(text_key),2)) '\n']);
			end
		end

		function plot_result(obj, clusters, title_text, plot_n)
			classified = clusters('classification');
			next_mean = clusters('mean_matrix');
			seed_mean = clusters('seed_mean');

			if (max(classified) > 6)
				return
			end
	

			max_range = max(max(obj.x));
			min_range = min(min(obj.x));

			subplot(plot_n,1,1);
			hold on;
			for m = 1:obj.n
				if(classified(m) == 1)
					plot(obj.x(1,m), obj.x(2,m),'ro');
				elseif(classified(m) == 2)
					plot(obj.x(1,m), obj.x(2,m),'go');
				elseif(classified(m) == 3)
					plot(obj.x(1,m), obj.x(2,m),'bo');
				elseif(classified(m) == 4)
					plot(obj.x(1,m), obj.x(2,m),'yo');
				elseif(classified(m) == 5)
					plot(obj.x(1,m), obj.x(2,m),'co');
				else
					plot(obj.x(1,m), obj.x(2,m),'mo');
				end
			end

			plot( seed_mean(1,:), seed_mean(2,:),'k^');
			plot( next_mean(1,:), next_mean(2,:),'kx');
			axis([min_range-1,max_range+1,min_range-1,max_range+1]);			
			title(title_text);
			hold off;

			subplot(plot_n,1,2);
			plot(clusters('error_convergence')); title('SSE');

			if clusters.isKey('log_likelihood_list')
				subplot(plot_n,1,3);
				plot(clusters('log_likelihood_list')); title('Likelihood');
			end

			%obj.print_outcome(clusters, 'Seed mean' , 'seed_mean');
			%obj.print_outcome(clusters, 'Mean' , 'mean_matrix');
			%obj.print_outcome(clusters, 'Cov Matrix' , 'cov_matrix');
			%obj.print_outcome(clusters, 'Prior' , 'prior');
		end
	end
end
