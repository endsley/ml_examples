#!/usr/bin/octave


cv = []
for k = 1:100
	n = 20; %floor(40*rand());
	d = floor(5*rand());
	A = zeros(d,d);
	p = floor(5*rand())*randn(n,d);
	for m = 1:n
		for n = 1:n
			v = p(m,:) - p(n,:);
			A = A + v'*v;
		end
	end
	
	[U,S,V] = svd(A);
	D = diag(S) + 1;

	%max(D)/min(D)
	cv = [cv max(D)/min(D)];
	
end

plot(cv)
xlabel('iteration')
ylabel('conditional value')
title('randomize sample size, variance, dimension for condition value with 100 iterations')
