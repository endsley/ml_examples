
function plot_2d_data(assignment, data, N, figure_id)
	dot_type = 'o';
	cmap = colormap('default');

	figure(figure_id);
	hold on;
	for m = 1:N
		if(assignment(m) == 1)
			%printf('plot 1\n')
			plot(data(1,m), data(2, m),['r' dot_type]);
		elseif(assignment(m) == 2)
			%printf('plot 2\n')
			plot(data(1,m), data(2, m),['g' dot_type]);
		elseif(assignment(m) == 3)
			%printf('plot 3\n')
			plot(data(1,m), data(2, m),['b' dot_type]);
		elseif(assignment(m) == 4)
			%printf('plot 4\n')
			plot(data(1,m), data(2, m),['m' dot_type]);
		elseif(assignment(m) == 5)
			%printf('plot 5\n')
			plot(data(1,m), data(2, m),['k' dot_type]);
		elseif(assignment(m) == 6)
			plot(data(1,m), data(2, m),['c' dot_type]);
		elseif(assignment(m) == 7)
			plot(data(1,m), data(2, m),'Color', cmap(5,:));
		elseif(assignment(m) == 8)
			plot(data(1,m), data(2, m),'Color', cmap(20,:));
		elseif(assignment(m) == 9)
			plot(data(1,m), data(2, m),'Color', cmap(59,:));
		elseif(assignment(m) == 10)
			plot(data(1,m), data(2, m),'Color', cmap(63,:));
		end
	end
	num_of_clusters = length(unique(assignment));
	title([num2str(num_of_clusters) ' clusters were found']);
	print(['Figure_' num2str(figure_id) '.png'])
end
