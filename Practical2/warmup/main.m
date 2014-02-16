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
% Part 0 - Plot initial data
%

fprintf('Plotting Data...\n')
figure(1)
plot(X,Y,'o');
xlabel('time since impact (ms)')
ylabel('g-force on head')
title('Initial Data')

%
% Part 1 - Calculate and plot via gradient descent
%

% compute via gradient descent
X = [ones(m, 1), input(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 50;
alpha = 0.001;

% compute initial cost
J = computeCostTheta(X, Y, theta);

fprintf('Press enter calculate and plot gradient descent.\n');
pause;

% run gradient descent
theta = gradientDescent(X, Y, theta, alpha, iterations);

fprintf('Final gradient descent cost is %f\n', computeCostTheta(X, Y, theta));

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-', 'color','r')
legend('Training data', 'Linear regression')
title('Linear Regression');
hold off % don't overlay any more plots on this figure

%
% Part 2 - Calculate and plot via polyfit
%
fprintf('Press enter to calculate polycost.\n');
pause;

% reset X to remove bias column
X = input(:,1);

p = polyfit(X,Y,9);
f = polyval(p,X);

figure(2)
plot(X,Y,'o',X,f,'-','color','r')
legend('Training data', 'Polyfit')
title('Polyfit');

sqrErrors = (f-Y).^2;
J = 1/(2*m) * sum(sqrErrors);

fprintf('polyerror = %f\n',J);
