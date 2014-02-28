function [r mu] = kmeanspp(X, K)
% Perform clustering with K-Means++.
%
% [r mu] = kmeanspp(X, K)
%
% Inputs:
%   X: an NxD matrix of N data with D dimensions
%   K: the number of clusters

% Outputs:
%   r:  an Nx1 vector of cluster assignments, integers in 1..K.
%   mu: a KxD matrix of the K cluster centers.
%
% Copyright Ryan P. Adams, 2014.
%

  % Get the dimensions of the data.
  [N D] = size(X);
  
  % Allocate memory for the cluster centers.
  mu = zeros([K D]);
	
  % First cluster center is a random datum.
  mu(1,:) = X(randi(N),:);
  
  % Loop over the remaining centers.
  for kk=2:K

    % Compute the minimum distances to existing clusters.
    min_dists2 = min(sqdist(X, mu(1:kk-1,:)),[],2);
    
    % Add a little bit to deal with potential underflow.
    min_dists2 = min_dists2 + 1e-6;
    
    % Normalize the squared distances into a distribution, then
    % draw from the resulting multinomial distribution over data.
    % Cast that into something that lets us slice easily.
    nn = logical(mnrnd(1, min_dists2 / sum(min_dists2)));
    
    % Set that datum as the next center.
    mu(kk,:) = X(nn',:);
    
  end

  % Get the final distanes.
  dists2 = sqdist(X, mu);
  
  % Compute the final assignments.
  [tmp r] = min(dists2, [], 2);
	
end