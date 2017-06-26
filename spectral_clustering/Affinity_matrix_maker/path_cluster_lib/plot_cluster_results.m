
function plot_cluster_results(x, assignment, data, N, figure_id)
	dot_type = '';

	figure(figure_id);
	hold on;
	for m = 1:N
		if(assignment(m) == 1)
			%printf('plot 1\n')
			plot(x, data(:, m),['r' dot_type]);
		elseif(assignment(m) == 2)
			%printf('plot 2\n')
			plot(x, data(:, m),['g' dot_type]);
		elseif(assignment(m) == 3)
			%printf('plot 3\n')
			plot(x, data(:, m),['b' dot_type]);
		elseif(assignment(m) == 4)
			%printf('plot 4\n')
			plot(x, data(:, m),['k' dot_type]);
		elseif(assignment(m) == 5)
			%printf('plot 5\n')
			plot(x, data(:, m),['y' dot_type]);
		elseif(assignment(m) == 6)
			plot(x, data(:, m),['c' dot_type]);
		end
	end
	num_of_clusters = length(unique(assignment));
	title([num2str(num_of_clusters) ' clusters were found']);
	print(['Figure_' num2str(figure_id) '.png'], '-S600,600')
	
end
