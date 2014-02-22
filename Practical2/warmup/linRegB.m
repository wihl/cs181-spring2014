% read in data
input = csvread('motorcycle.csv' );
X = input(:,1);
Y = input(:,2);
trainingPoints = length(Y); % number of training examples

a0 = -0.3; %Parameters of the actual underlying model that we wish to recover
a1 = 0.5;  %We will estimate these values with w0 and w1 respectively. 

noiseSD = 0.2;

priorPrecision = 2.0;

likelihoodSD = noiseSD;
likelihoodPrecision = 1/(likelihoodSD)^2;

xtrain = X;
ytrain = Y;

model = struct('mu', 0, 'Sigma', noiseSD);
noise = gaussSample(model, trainingPoints);

iter = 30;

%subplot2(iter+2,2,1,2);
priorMean = [0;0];
priorSigma = eye(2)./priorPrecision; %Covariance Matrix
priorPDF = @(W)gaussProb(W,priorMean',priorSigma);
%contourPlot(priorPDF,[]);

% Plot sample lines whose parameters are drawn from the prior distribution.
%subplot2(iter+2,2,1,3);
%plotSampleLines(priorMean',priorSigma,6,input)

% For each iteration plot the likelihood of the ith data point, the
% posterior over the first i data points and sample lines whose
% parameters are drawn from the corresponding posterior. 
mu = priorMean;
sigma = priorSigma;


for i=1:iter
  %subplot2(2+iter,3,i+1,1);
  likelihood = @(W) uniGaussPdf(xtrain(i),W*[1;xtrain(i)],likelihoodSD.^2);
 % contourPlot(likelihood,[a0,a1]);
  
  %subplot2(2+iter,3,i+1,2);
  [postW,mu,sigma] = update([1,xtrain(i)],ytrain(i),likelihoodPrecision,mu,sigma);
  %contourPlot(postW,[a0,a1]);
  
  %subplot2(2+iter,2,i+1,3);
  %plotSampleLines(mu,sigma,6,[xtrain(1:i),ytrain(1:i)]);  

  % plot every tenth iteration
  if mod(i,10) == 0
     subplot(3,1,floor(i/10));
     plotSampleLines(mu, sigma, 20, input);
  endif
end

