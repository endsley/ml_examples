
clear all;clc;

warning ('off','all');
addpath('src');
addpath('mutual_information');
format short

load ./data/alternative_truth.mat;
load ./data/Ground_truth.mat;
Y = double(Y);
G = double(G);
k = 20;
k2 = 4;

files = {};
	files{1} = './data/face_fft.mat';
	files{2} = './data/face_gabor.mat';  
	files{3} = './data/face_hog.mat';
	files{4} = './data/face_lbp.mat';
	files{5} = './data/face_pca.mat';
	files{6} = './data/full_data.mat';
    
for m = 1:length(files)
	load(files{m})
	fprintf('file : %s\n', files{m});

	[assignment_1, assignment_2, center_1, center_2] = constrained_alternative_clustering(k, k2, X);

	face_fft_1 = nmi(assignment_1(:),G(:));
	face_fft_2 = nmi(assignment_2(:),Y(:));
	fprintf('\t%f , %f\n', face_fft_1, face_fft_2);

	%fprintf('\t%f\n', face_fft_1);
end




%
%
%addpath('code_Metric_online')
%load clust_5.mat
%
%
%%	Run initial K means
%	k = 2;
%	opt_error_d = -1;
%	opt_idx = 0;
%	for m = 1:10
%		[idx,C,sumd] = kmeans(X,k);
%		error_d =norm(sumd)
%		if(opt_error_d == -1)
%			opt_error_d = error_d;
%			opt_idx = idx;
%		end
%	
%		if(error_d < opt_error_d)
%			opt_idx = idx;
%			opt_error_d = error_d;
%		end
%	end
%	
%%	plot_kmeans(X, opt_idx);
%
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
%
%%	Plot results
%	subplot(3,1,1);
%	plot(X(1:3,1),X(1:3,2),'bx'); hold on;
%	plot(X(4:6,1),X(4:6,2),'gx'); 
%	plot(X(7:9,1),X(7:9,2),'kx'); 
%	plot(X(10:12,1),X(10:12,2),'mx'); 
%	title('Original Data')
%
%
%	subplot(3,1,2);
%	plot(y(1:3,1),y(1:3,2),'bx'); hold on;
%	plot(y(4:6,1),y(4:6,2),'gx'); 
%	plot(y(7:9,1),y(7:9,2),'kx'); 
%	plot(y(10:12,1),y(10:12,2),'mx'); 
%	axis([-1,0.5,-1,0.5]);
%	title('After Distance Matrix Transformation')
%
%	subplot(3,1,3);
%	plot(y2(1:3,1),y2(1:3,2),'bx'); hold on;
%	plot(y2(4:6,1),y2(4:6,2),'gx'); 
%	plot(y2(7:9,1),y2(7:9,2),'kx'); 
%	plot(y2(10:12,1),y2(10:12,2),'mx'); 
%	axis([-200,200,-1400,200.5]);
%	title('After Alternative Distance Matrix Transformation')
%
