
%	Each column in A is a single data point
function out_matrix = get_Distance_in_Time(A, weight, remove_percentage, tight_bound )
	if weight == 0
		out_matrix = [1];
		return
	end
	A = cell2mat(A(:));
	A = variance_map_filter(A, remove_percentage);

	N = size(A,2);
	Euclid_matrix = [];
	for m = 1:N
		D = abs(A - repmat(A(:,m), 1, N));
		single_row = norm(D,2,'columns');
		Euclid_matrix = [Euclid_matrix;single_row];
	end

	if(tight_bound == 1)
		save -6 Euclid.mat Euclid_matrix
		system('./set_boundary.py')
		out_matrix = 1;
	else
		sorted_M = sort(Euclid_matrix);
		%K_closest = sorted_M(floor(N/2),:)';			% using the median
		K_closest = sorted_M(7,:)';					% using the 7th closest approach
		out_matrix = (Euclid_matrix.^2)./(K_closest*K_closest');
	end

%	figure(32);
%	idd = 67;
%	Adjacency_matrix = exp(-out_matrix);
%	subplot(2,1,1);
%	plot(Euclid_matrix(idd,:), Adjacency_matrix(1,:), 'o')
%	subplot(2,1,2);
%	hist(Euclid_matrix(idd,:),40)

%format long g;
%	ee = Euclid_matrix.^2;
%	hist(ee(:,100),30);
%exp(-ee(:,100)./0.01)
%	K_closest(3)

end
