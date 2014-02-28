function Y = standardize(X)
% Make each dimension have zero mean and unit variance.
%
% Y = standardize(X)
%
% Inputs:
%  X: NxD matrix of N data vectors in D dimensions
%
% Outputs:
%  Y: NxD matrix of N standardized data vectors.
%
% Copyright Ryan P. Adams, 2014.
%
  
  means = mean(X,1);
  stds  = std(X,1);
  Y = bsxfun(@rdivide, bsxfun(@minus, X, means), stds);
end
