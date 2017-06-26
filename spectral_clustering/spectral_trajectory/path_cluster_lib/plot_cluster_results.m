
%function plot_cluster_results(x, assignment, data, N, figure_id)
%	dot_type = '';
%
%	figure(figure_id);
%	hold on;
%	for m = 1:N
%		if(assignment(m) == 1)
%			%printf('plot 1\n')
%			plot(x, data(:, m),['r' dot_type]);
%		elseif(assignment(m) == 2)
%			%printf('plot 2\n')
%			plot(x, data(:, m),['g' dot_type]);
%		elseif(assignment(m) == 3)
%			%printf('plot 3\n')
%			plot(x, data(:, m),['b' dot_type]);
%		elseif(assignment(m) == 4)
%			%printf('plot 4\n')
%			plot(x, data(:, m),['k' dot_type]);
%		elseif(assignment(m) == 5)
%			%printf('plot 5\n')
%			plot(x, data(:, m),['y' dot_type]);
%		elseif(assignment(m) == 6)
%			plot(x, data(:, m),['c' dot_type]);
%		elseif(assignment(m) == 7)
%			plot(x, data(:, m),'color', [0.5 0.3 0.2]);	
%		elseif(assignment(m) == 8)
%			plot(x, data(:, m),'color', [1 0 1]);	
%		end
%	end
%	num_of_clusters = length(unique(assignment));
%	title([num2str(num_of_clusters) ' clusters were found']);
%	print(['Figure_' num2str(figure_id) '.png'])
%end


function plot_cluster_results(x, assignment, data, N, figure_id)
	dot_type = '';

	num_of_clusters = length(unique(assignment));
	for n = 1:num_of_clusters
		figure(n);
		hold on;
		counter = 0;
		for m = 1:N
			if(assignment(m) == n)
				%printf('plot 1\n ')
				plot(x, data(:, m),['r' dot_type]);
				counter = counter + 1;
			end
		end
		
		title([num2str(num_of_clusters) 'clusters with ' num2str(counter) ' samples']);
		print(['gene_1_cluster_' num2str(n) '.png'])
	end
end
