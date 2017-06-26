
%	Each column in A is a single data point
function divergence_matrix = calc_Jensen_shannon_divergence(A)
	N = size(A,2);
	divergence_matrix = [];

	for m = 1:N
		v = repmat(A(:,m), 1, N);
		single_row = sum(A.*log(A./v));
		divergence_matrix = [divergence_matrix;single_row];
	end
	divergence_matrix = (divergence_matrix + divergence_matrix');
end
