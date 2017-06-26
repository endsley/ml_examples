
% mutual information = sum p(x,y) log2[ p(x,y) / (p(x)*p(y) ] 
%	Note : this is base 2

function [mInfo, xkey, ykey, x_prob, y_prob, x_y_prob] = calc_mutual_information(x,y)
	if length(x) ~= length(y)
		error('The length of two vectors must be equal to calculate the mutual information')
	end

	xkey = unique(x, 'rows');
	ykey = unique(y, 'rows');

	x_prob = ones(1, length(xkey));
	y_prob = ones(1, length(ykey));
	x_y_prob = ones(length(xkey), length(ykey));

	%	Calculate Prob of each outcome
	for m = 1:length(xkey)
		x_prob(m) = sum(xkey(m) == x);
	end
	for m = 1:length(ykey)
		y_prob(m) = sum(ykey(m) == y);
	end
	x_prob = x_prob/length(x);
	y_prob = y_prob/length(y);


	for m = 1:length(xkey)
		for n = 1:length(ykey)
			x_y_prob(m,n) = sum((xkey(m) == x) & (ykey(n) == y));
			%printf('%d %d\n', xkey(m), ykey(n));
		end
	end
	x_y_prob = x_y_prob/sum(sum(x_y_prob));

	independent_join_pdf =  x_prob'*y_prob;
	tmp = log2(x_y_prob./independent_join_pdf);

	mInfo = sum(x_y_prob(tmp ~= -Inf).*tmp(tmp ~= -Inf));

end
