%#!/usr/bin/octave

addpath('code_Metric_online')
load clust_5.mat


%	Run initial K means
	k = 2;
	opt_error_d = -1;
	opt_idx = 0;
	opt_C = 0;
	for m = 1:10
		[idx,C,sumd] = kmeans(X,k);
		error_d =norm(sumd)
		if(opt_error_d == -1)
			opt_error_d = error_d;
			opt_idx = idx;
			opt_C = C;
		end
	
		if(error_d < opt_error_d)
			opt_idx = idx;
			opt_error_d = error_d;
			opt_C = C;
		end
	end
	
%	plot_kmeans(X, opt_idx);

	
	[y,y2] = gen_alter_clusters(X, k)

%%	Create the must link / cannot link matrices
%	S = zeros(length(X));
%	D = zeros(length(X));
%	for n = 1:k
%		vec = double(opt_idx == n);
%		S = S + vec*vec';
%	end
%	D = abs(S-1)
%
%%	Generate Distance matrix and transform the data
%	A = opt(X, S, D, 20)
%	[U,S,V] = svd(A);
%	y = X*U*sqrt(S);
%	y2 = X*U*inv(sqrt(S));

%	Plot results
	subplot(3,1,1);
	plot(X(1:3,1),X(1:3,2),'bx'); hold on;
	plot(X(4:6,1),X(4:6,2),'gx'); 
	plot(X(7:9,1),X(7:9,2),'kx'); 
	plot(X(10:12,1),X(10:12,2),'mx'); 
	title('Original Data')


	subplot(3,1,2);
	plot(y(1:3,1),y(1:3,2),'bx'); hold on;
	plot(y(4:6,1),y(4:6,2),'gx'); 
	plot(y(7:9,1),y(7:9,2),'kx'); 
	plot(y(10:12,1),y(10:12,2),'mx'); 
	axis([-1,0.5,-1,0.5]);
	title('After Distance Matrix Transformation')

	subplot(3,1,3);
	plot(y2(1:3,1),y2(1:3,2),'bx'); hold on;
	plot(y2(4:6,1),y2(4:6,2),'gx'); 
	plot(y2(7:9,1),y2(7:9,2),'kx'); 
	plot(y2(10:12,1),y2(10:12,2),'mx'); 
	axis([-200,200,-1400,200.5]);
	title('After Alternative Distance Matrix Transformation')

