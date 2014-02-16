%
% CSCI-E181, Spring 2014, Practical 2, Warmup
% David Wihl
% 
% Usage:
%     Start Octave or Matlab
%     octave> main
%


% read in data
input = csvread('motorcycle.csv' );
X = input(:,1);
Y = input(:,2);
m = length(Y); % number of training examples

%
% Part 1 - Calculate and plot via gradient descent
%

fprintf('Press enter to calculate and plot gradient descent.\n');
pause;

% compute via gradient descent
X_bias = [ones(m, 1), input(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 50;
alpha = 0.001;

% run gradient descent
theta = gradientDescent(X_bias, Y, theta, alpha, iterations);

fprintf('Final gradient descent cost is %f\n', computeCost(X_bias, Y, X_bias*theta));

plotRegression(X,Y,X_bias*theta,'Linear Regression')

%
% Part 2 - Calculate and plot via polyfit
%
fprintf('Press enter to calculate polycost.\n');
pause;

% find the best polynomial fit, up to 12
bestp = 1;
bestJ = Inf;
for i = 1:12
  p = polyfit(X,Y,i);
  f = polyval(p,X);
  J = computeCost(X,Y,f);
  if J < bestJ
     bestJ = J;
     bestp = i;
  endif

end

plotRegression(X,Y,f,'Polyfit')
