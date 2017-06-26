
function [y_total, y_normalized] = generate_distribution(N, sigma, sigma_2, x1,y1)
	epsilon = 0.000001;

	%	Create original y
	A = [ones(length(x1),1) x1 x1.^2 x1.^3 x1.^4 x1.^5 x1.^6];
	[q r] = qr(A);
	coef = r\(q'*y1);

	x_lower = min(x1);
	x_upper = max(x1);

	x = [0:99]';
	A = [ones(length(x),1) x x.^2 x.^3 x.^4 x.^5 x.^6];
	y = A*coef;

	y((x<x_lower) | (x>x_upper)) = 0;
	y_total = [y];

	%	Create noisy y
	for m = 1:N
		x2 = x1 + sigma*randn(length(x1),1);
		y2 = y1 + sigma*randn(length(y1),1);

		A = [ones(length(x2),1) x2 x2.^2 x2.^3 x2.^4 x2.^5 x2.^6];
		[q r] = qr(A);
		coef = r\(q'*y2);
	
		x = [0:99]';
		A = [ones(length(x),1) x x.^2 x.^3 x.^4 x.^5 x.^6];
		y = A*coef;
		y = y + sigma_2*randn(length(y),1);

		y((x<x_lower) | (x > x_upper)) = 0;
		y_total = [y_total, y];

	end

	y_total(y_total <= 0) = epsilon;
	y_normalized = y_total./repmat(sum(y_total),size(y_total,1),1);
end
