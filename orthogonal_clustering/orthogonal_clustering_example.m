
clear all;clc;

warning ('off','all');
addpath('src');
addpath('mutual_information');
format short

load alternative_truth.mat;
load Ground_truth.mat;
Y = double(Y);
G = double(G);
k = 20;
k2 = 4;

files = {};
	files{1} = 'face_fft.mat';
	files{2} = 'face_gabor.mat';  
	files{3} = 'face_hog.mat';
	files{4} = 'face_lbp.mat';
	files{5} = 'face_pca.mat';
	files{6} = 'full_data.mat';
    
%for m = 1:length(files)
%	load(files{m})
%	fprintf('file : %s\n', files{m});
%		
%	[assignment_1, assignment_2, center_1, center_2] = orthogonal_cluster(k, k2, X);
%	face_fft_1 = nmi(assignment_1(:),G(:));
%	face_fft_2 = nmi(assignment_2(:),Y(:));
%
%	fprintf('\t%f , %f\n', face_fft_1, face_fft_2);
%end

	load(files{0})
	fprintf('file : %s\n', files{0});
		
	[assignment_1, assignment_2, center_1, center_2] = orthogonal_cluster(k, k2, X);
	face_fft_1 = nmi(assignment_1(:),G(:));
	face_fft_2 = nmi(assignment_2(:),Y(:));

	fprintf('\t%f , %f\n', face_fft_1, face_fft_2);


