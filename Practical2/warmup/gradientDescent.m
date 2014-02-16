function theta = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples

for iter = 1:num_iters

    theta = theta - ((X*theta-y)'*X)'*alpha/m;
    % fprintf('theta: %f\n', theta);

end

end
