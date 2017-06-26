#!/usr/bin/octave

% This script is used to test various K Finding approaches on a
% one-dimensional vecotr

clear all; close all; clc;

% Turn off warnings
warning('off');
% Load dataset and extract data of interest
load Euclid;
data = Euclid_matrix(:,101);

%% Automatically discover the number of clusters
% Specify the search range of K
K_range = 1:10;
% Specify the candidate clustering algorithms
alg_clust_list = {'kmeans','linkage','gmdistribution'};
% Specify the criterion to find K
criteria_list = {'gap','silhouette','DaviesBouldin','CalinskiHarabasz'};

% iteration over clustering algorithms
for idx_alg_clust = 1:length(alg_clust_list)
    alg_clust = alg_clust_list{idx_alg_clust};
    % iteration over clustering criterion
    for idx_criteria = 1:length(criteria_list)
        criteria = criteria_list{idx_criteria};
        disp('=============================================================');
        disp(['clustering algorithm:  ',alg_clust,';    criteria:  ',criteria]);
        eva = evalclusters(data,alg_clust,criteria,'KList',K_range);
        disp(['OptimalK = ',num2str(eva.OptimalK)]);
    end
    
end
