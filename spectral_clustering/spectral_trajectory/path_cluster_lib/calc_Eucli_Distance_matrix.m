
%	Each column in A is a single data point
function Euclid_matrix = calc_Eucli_Distance_matrix(A, use_L1)
	N = size(A,2);
	Euclid_matrix = [];
	for m = 1:N
		if(use_L1 == 1)
			single_row = sum(abs(A - repmat(A(:,m), 1, N)));
		else
			D = abs(A - repmat(A(:,m), 1, N));
			if(l == 1)
				single_row = sqrt(D.^2);
			else
				single_row = sqrt(sum(D.^2));
			end
		end
		Euclid_matrix = [Euclid_matrix;single_row];
	end

end
