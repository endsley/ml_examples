
function kernel = get_kernel(W, sigma)
	output_precision(5);

	m = csvread('./data_1.csv');
	N = size(m,1);

	kernel = zeros(N,N);

	for i = 1:N
		for j = 1:N

			ith = m(i,:);
			jth = m(j,:);
			
			d = ith - jth;
			kernel(i,j) = exp((-d*W*W'*d(:))/(2*sigma));
			kernel(i,j) = round(kernel(i,j)*10000)/10000;

			%printf('%d , %d : \n', i, j)
			%kernel(i,j)

		end
	end

end
