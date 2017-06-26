#!/usr/bin/octave

p = 0.1:0.1:0.9;
q = 1-p;

x = [p;q];

%tx = ty = linspace (-8, 8, 41)';
%[xx, yy] = meshgrid (tx, ty);
%r = sqrt (xx .^ 2 + yy .^ 2) + eps;
%tz = sin (r) ./ r
%mesh (tx, ty, tz);

distance_matrix = calc_Jensen_shannon_divergence(x)
Euclid_matrix = calc_Eucli_Distance_matrix(x)



	N = size(x,2);
	divergence_matrix = [];

	for m = 1:N
		v = repmat(x(:,m), 1, N);
		single_row = sum(x.*log10(x./v));
		divergence_matrix = [divergence_matrix;single_row];
	end
	divergence_matrix = (divergence_matrix + divergence_matrix');



rg = -0.4:0.1:0.4;
plot(rg, diag(fliplr(distance_matrix)));
hold on;
plot(rg, diag(fliplr(divergence_matrix)),'g');
plot(rg, diag(fliplr(Euclid_matrix)),'r');

title('KL divergence vs Euclidean kernel of Binomial Distribution')
xlabel('Functional deviation')
ylabel('Distance')

%mesh (p, q, distance_matrix);
input('')
