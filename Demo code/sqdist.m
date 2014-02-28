function D2 = sqdist(X, Y)
% Compute the squared distances between two sets of vectors.
%
% D2 = sqdist(X, Y)
%
% Inputs:
%   X: an NxD matrix of vectors
%   Y: an MxD matrix of vectors
%
% Outputs:
%   D2: an NxM matrix of squared distances between all NM pairs.
%
% Copyright Ryan P. Adams, 2014
%
 
  D2 = bsxfun(@plus, sum(X.^2,2), sum(Y.^2,2)') - 2*X*Y';
end