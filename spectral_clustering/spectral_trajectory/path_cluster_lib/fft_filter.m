
function out_matrix = fft_filter(A, reduction_percentage)
	if reduction_percentage == 0
		out_matrix = A;
		return;
	end

	dat_size = size(A,1);
	increments = floor(reduction_percentage*( dat_size - 1 )/2);
	first = ceil((dat_size - 1)/2) + 1 - increments
	second = ceil((dat_size - 1)/2 + 0.5) + 1 + increments

	f = fft(A);
	f(first:second,:) = 0;
	out_matrix = abs(ifft(f));
end
