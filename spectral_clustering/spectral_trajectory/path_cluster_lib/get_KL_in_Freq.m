
%	Each column in A is a single data point
function out_matrix = get_KL_in_Freq(A)
	A = abs(fft(A));

	N = size(A,2);
	divergence_matrix = [];

	for m = 1:N
		v = repmat(A(:,m), 1, N);
		single_row = sum(A.*log(A./v));
		divergence_matrix = [divergence_matrix;single_row];
	end
	divergence_matrix = (divergence_matrix + divergence_matrix');
%	K_closest = median(divergence_matrix)

sorted_M = sort(divergence_matrix);
%sorted_M
%sorted_M
%max(divergence_matrix(:))
%sorted_M

	K_closest = sorted_M(floor(size(A,1)/2),:)';
	%K_closest = sorted_M(7,:)';

	out_matrix = (divergence_matrix.^2)./(K_closest*K_closest');
end
