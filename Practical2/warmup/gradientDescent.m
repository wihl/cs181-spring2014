function theta = gradientDescent(X, Y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta

m = length(Y); % number of training examples

for iter = 1:num_iters

    theta = theta - ((X*theta-Y)'*X)'*alpha/m;
%    fprintf("Iteration: %d, J = %f, theta0 = %f, theta1 = %f\n",
%	    iter, computeCost(X,Y,X*theta), theta(1), theta(1))

    % plot(X,X*theta,'-','color','m');


end

end
