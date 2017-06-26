

function [assignment_1, assignment_2, center_1, center_2] = orthogonal_cluster(k, k2, X)
	kmeans_repeat = 10;
	addpath('cbrewer');

	X = X - repmat(mean(X),size(X,1),1);
	[colormap]=cbrewer('qual', 'Accent', k, 'cubic');

	%	Run initial K means 10 times and keeps the best result
	[assignment_1 , center_1] = pack_kmeans(X,k, kmeans_repeat);
	%figure(1); plot_kmeans(X, assignment_1, colormap);
	
	
	%	Find projected new_X
	new_X = [];
	for m = 1:size(X,1)
		cluster_id = assignment_1(m);

		v = center_1(cluster_id,:)';
		x = X(m,:)';

		proj_x = v*((v'*x)/(v'*v));
		orth_x = x - proj_x;

		new_X = [new_X; orth_x'];
	end

	[assignment_2 , center_2] = pack_kmeans(new_X, k2, kmeans_repeat);
	%figure(2); plot_kmeans(X, assignment_2, colormap);
	
end

