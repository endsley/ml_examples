function obj = evalclusters(X, YorFun, crit,varargin)
%EVALCLUSTERS Evaluate clustering solutions.
%   EVA = EVALCLUSTERS(X, CLUST, CRITERION) creates an evaluation object
%   which can be used for estimating the number of clusters on data. 
%
%   X must be an N-by-P matrix of data with one row per observation and one
%   column per variable. CLUST represents a clustering algorithm or a set
%   of clustering solutions. CRITERION is a string representing the
%   criterion to be used. The value of CRITERION could be 'CalinskiHarabasz',
%   'Silhouette', 'gap' or 'DaviesBouldin'.
%
%   EVA is an object of a particular class based on the value of CRITERION:
%               Value              CLASS
%              'CalinskiHarabasz'  CalinskiHarabaszEvaluation
%              'Silhouette'        SilhouetteEvaluation
%              'gap'               GapEvaluation
%              'DaviesBouldin'     DaviesBouldinEvaluation
%
%   CLUST must be a string, a function handle, or a numeric matrix.
%      If CLUST is a string, it specifies one of the following built-in
%      clustering algorithms:
%         'kmeans':  
%              Function KMEANS will be used, with the 'EmptyAction'
%              option being 'singleton' and the 'Replicates' value 
%              being 5.
%         'linkage': 
%              Function CLUSTERDATA will be used, with the 'Linkage' value
%              being 'ward' by default.
%         'gmdistribution':  
%              FITGMDIST will be used, with the 'SharedCov' value
%              being true and the 'Replicates' value being 5.
%     
%      If CLUST is a function handle specified using @, such as @CLUSTFUN,
%      it must be a handle to a function of the form:
%  
%               function C = CLUSTFUN(DATA, K)
%  
%      where DATA is the data to be clustered, and K is the number of
%      clusters. The output of CLUSTFUN must be one of the following:
%         A vector of integers representing the cluster index for each
%         observation in DATA. The number of the unique values in this
%         vector must be K.
%         A N-by-K numeric matrix of scores for N observations and
%         K clusters. In this case, the cluster index for each observation
%         is decided by taking the largest score value in each row.
%              
%      If CLUST is a matrix, it represents a set of clustering solutions.
%      It must have N rows and contain integers. Column J contains the
%      cluster indices for each of the N points in the Jth clustering
%      solution. CLUST cannot be a set of clustering solutions when
%      CRITERION is 'gap'.
%
%   EVA = EVALCLUSTERS(..., 'NAME1',VALUE1,...,'NAMEN',VALUEN) accepts one
%   or more comma-separated optional argument name/value pairs.
%    
%   For all the criteria:
%      'KList'    - A vector of positive integers indicating the number of
%                   clusters to be evaluated. It must be provided when
%                   CLUST is a string or a function handle representing a
%                   clustering algorithm. 
%    
%   For Criteria 'Silhouette' and 'gap':
%      'Distance' - The distance measure used for clustering and for
%                   computing the 'Silhouette' or the 'Gap' values.
%                   The value of distance can be:
% 
%          'sqEuclidean'  - Squared Euclidean distance (default)
%          'Euclidean'    - Euclidean distance
%          'cityblock'    - Sum of absolute differences, a.k.a. L1
%          'cosine'       - One minus the cosine of the included angle
%                           between points (treated as vectors)
%          'correlation'  - One minus the sample correlation between
%                           points (treated as sequences of values)
%          'Hamming'      - Percentage of coordinates that differ (only
%                           valid for the 'Silhouette' criterion.)
%          'Jaccard'      - Percentage of non-zero coordinates that differ
%                           (only valid for the 'Silhouette' criterion.)
%          vector         - A numeric distance matrix in the vector form
%                           created by PDIST (only valid for the
%                           'Silhouette' criterion.)
%          function       - A distance function specified using @, for
%                           example @DISTFUN
%
%          A distance function must be of the form
%
%             function D2 = DISTFUN(XI, XJ),
%
%          taking as arguments a 1-by-P vector XI containing a single row
%          of X, an M2-by-P matrix XJ containing multiple rows of X, and
%          returning an  M2-by-1 vector of distances D2, whose Jth element
%          is the distance between the observations XI and XJ(J,:).
%
%      The choice of distance measure should generally match the distance
%      measure used in the clustering algorithm to obtain meaningful
%      results. When CLUST is a 'kmeans', the clustering is automatically
%      based on the selected distance measure. When CLUST is 'linkage', the
%      clustering depends on the selected distance measure as follows:
% 
%             Selected          distance used        linkage used
%            'Distance'         for clustering       for clustering
%            ----------         --------------       --------------
%            'sqEuclidean'      Euclidean            Ward
%            'Euclidean'        Euclidean            Ward
%            any other          selected metric      Average
%
%   For CRITERION 'Silhouette':
%     'ClusterPriors' -  Prior probabilities for each cluster.
%                'empirical':    The silhouette value for each clustering
%                                solution is computed as the average of
%                                the silhouette values among all the
%                                points. (Default)
%                'equal':        The criterion value for each clustering
%                                solution is computed as the mean of
%                                within-cluster average silhouette values.
%   
%   For Criterion 'gap':
%     'B'             -   A positive integer representing the number of
%                         reference data sets used for computing the gap
%                         criterion values. Default is 100.
%               
%     'ReferenceDistribution' - A string indicating how reference data are
%                               generated. 
%                 'uniform':     Generate each reference feature uniformly
%                                over the range of each feature of the
%                                input data.
%                 'PCA':         Generate the reference feature from a
%                                uniform distribution over a box aligned
%                                with the principal components of the input
%                                data. (Default)
%                  
%    'SearchMethod' - A string indicating how the optimal number of clusters
%                     are estimated. 
%                     'firstmaxse': The optimal number of clusters is the
%                                   smallest number of clusters, K,
%                                   satisfying:
%                                       GAP(K) >= GAP(K') - SE(K')
%                                   where GAP(K) is the gap value for
%                                   the clustering solution with K clusters.
%                                   K' is the next value in the sorted
%                                   KList. SE(K') is the standard error of
%                                   clustering solution with K' clusters.
%                     'globalmaxse':The optimal number of clusters is the 
%                                   smallest K satisfying 
%                                      GAP(K) >= GAPMAX-SE(GAPMAX)
%                                   where GAPMAX is the largest gap value,
%                                   and SE(GAPMAX) is the standard error
%                                   corresponding to this largest gap value.
%                     Default: 'globalmaxse'
%
%  Example: 
%     %Use the CalinskiHarabasz criterion to estimate the number of clusters
%     load fisheriris
%     E = evalclusters(meas,'kmeans','CalinskiHarabasz','klist',[1:5]);
%     %Evaluate another list of number of clusters
%     E = addK(E,[6:8]);
%     plot(E);
%
%  See also clustering.evaluation.GapEvaluation,
%           clustering.evaluation.SilhouetteEvaluation,
%           clustering.evaluation.CalinskiHarabaszEvaluation,
%           clustering.evaluation.DaviesBouldinEvaluation.

%   Copyright 2013 The MathWorks, Inc.

 narginchk(3,inf);

 [~,critPos] = internal.stats.getParamVal(crit,...
          {'CalinskiHarabasz', 'GAP', 'Silhouette', 'DaviesBouldin'},...
          'CRITERION');
  switch critPos
      case 1 %'CalinskiHarabasz'
           obj = clustering.evaluation.CalinskiHarabaszEvaluation(X,YorFun,varargin{:});
      case 2 %'GAP'
           obj = clustering.evaluation.GapEvaluation(X,YorFun,varargin{:});
      case 3 %'Silhouette'
           obj = clustering.evaluation.SilhouetteEvaluation(X,YorFun,varargin{:});
      case 4 %'DaviesBouldin'
           obj = clustering.evaluation.DaviesBouldinEvaluation(X,YorFun,varargin{:});
  end
          
      
