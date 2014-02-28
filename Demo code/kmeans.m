function [r mu] = kmeans(X, K, ppinit)
% Perform K-Means clustering via Lloyd's algorithm.
%
% [r mu] = kmeans(X, K, ppinit)
%
% This function takes two required arguments and one optional
% argument.  It initializes a clustering and then performs
% iterations of Lloyd's algorithm until it converges.  Convergence
% is determined by when the responsibilities stop changing.
%
% Inputs:
%   X:      an NxD matrix of N data with D dimensions
%   K:      the number of clusters
%   ppinit: a flag to indicate whether to initialize randomly (0) or
%           with K-Means++ (1).
%
% Outputs:
%   r:  an Nx1 vector of cluster assignments, integers in 1..K.
%   mu: a KxD matrix of the K cluster centers.
%
% Copyright Ryan P. Adams, 2014.
%

  if nargin == 2
    % If we only get two input arguments, assume that we should
    % initialize with K-Means++.
    ppinit = 1;
  end
  
  if ppinit == 1
    % Do the K-Means++ initialization.
    [r mu] = kmeanspp(X, K);
  else
    % Make random assignments of data to clusters.
    r  = randi(K, [size(X,1) 1]);
    mu = randn([K size(X,2)]);
  end
	
  % Track assignments from the previous iteration.
  last_r = zeros(size(r));
  
  % We're done when the assignments don't change.
  while sum(last_r ~= r) > 0
    fprintf('%d assignments changed.\n', sum(last_r ~= r));
    last_r = r;
    
    % Loop over the clusters and compute the means.
    for kk=1:K
      
      % Here I'm slicing out the data for this cluster and then
      % just computing their means directly.
      mu(kk,:) = mean(X(r==kk,:),1);
    end	
    
    % Compute the squared distances between data and centers.
    dists2 = sqdist(X, mu);
    
    % Get the indices of the minima, row-wise.
    [tmp r] = min(dists2, [], 2);		
  end	
  
end
