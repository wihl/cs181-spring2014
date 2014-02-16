function J = computeCost(X, Y, f)
%COMPUTECOST Compute cost for linear regression
m = length(Y); % number of training examples

J = 1/(2*m) * sum( (f - Y) .^ 2);

end
