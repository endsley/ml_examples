
%	Each column in A is a single data point
function out_matrix = get_Distance_in_Freq(A, weight, remove_percentage, tight_bound )
	if weight == 0
		out_matrix = [1];
		return
	end

	A_out{1} = 0;
	for m = 1:length(A)
		len = floor(size(A{m},1)/2);
		p = abs(fft(A{m}));
		p = p(1:len, :);
		p = variance_map_filter(p, remove_percentage);
		A_out{m} = p;
	end
	A = cell2mat(A_out(:));


	N = size(A,2);
	Euclid_matrix = [];
	for m = 1:N
		D = abs(A - repmat(A(:,m), 1, N));
		single_row = sqrt(sum(D.^2));
		Euclid_matrix = [Euclid_matrix;single_row];
	end

	if(tight_bound == 1)
		save -6 Euclid.mat Euclid_matrix
		system('./set_boundary.py')
		out_matrix = Euclid_matrix;
	else
		sorted_M = sort(Euclid_matrix);
		K_closest = sorted_M(floor(N/2),:)';
		out_matrix = (Euclid_matrix.^2)./(K_closest*K_closest');
	end
end
