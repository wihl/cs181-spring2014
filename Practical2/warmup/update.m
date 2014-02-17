%
% Given the mean = priorMu and covarianceMatrix = priorSigma of a prior
% Gaussian distribution over regression parameters; observed data, xtrain
% and ytrain; and the likelihood precision, generate the posterior
% distribution, postW via Bayesian updating and return the updated values
% for mu and sigma. xtrain is a design matrix whose first column is the all
% ones vector.
function [postW,postMu,postSigma] = update(xtrain,ytrain,likelihoodPrecision,priorMu,priorSigma)

postSigma  = inv(inv(priorSigma) + likelihoodPrecision*xtrain'*xtrain); 
postMu = postSigma*inv(priorSigma)*priorMu + likelihoodPrecision*postSigma*xtrain'*ytrain; 
postW = @(W)gaussProb(W,postMu',postSigma);

end
