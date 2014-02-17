function J = computeCost(X, T, y)
%COMPUTECOST Compute cost for linear regression
% use sum of squares.
% Inputs:
%   X - the input vector
%   T - the target values
%   y - the output function

m = length(T); % number of training examples

J = 1/(2*m) * sum( (y - T) .^ 2);

end
