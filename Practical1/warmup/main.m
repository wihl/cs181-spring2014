%
% main.m
% 
% usage:
%
%    1. start Octave
%    2. load the CIFAR-10 data
%         octave> load ('cifar-10-matlab/data_batch_1')
%    3. Set K
%         octave> K=10
%    4. Run this file
%         octave> main(data,K)
%
% The results will show a relatively useless plot of the centroids. More usefully, it will display
% the percentage of points in each cluster per iteration.
%




function [centroids, idx] = main(data, K)
close all; clc

setSize = 10; % number of points per generated cluster

if K < 1
   % default to a sane number
   K = 3;
endif

if ~exist('X', 'var')
% generate a random set X separated by 10's
    X = [];

    for i = 0:(K-1)
        X = cat(1, X, rand(setSize,2) + (i * 10));
        i = i + 1;
    end
endif

% normalize X. Comment out to leave unnormalized
X = double(data - mean(data)) ./ std(data);

% X = data;

% Uncomment to use preset centroids
% initial_centroids = [1,1;11,11;25,25;35,35;40,60]


if ~exist('initial_centroids', 'var')
    % Use K-Medoids - select from X
    initial_centroids = kMeansInitCentroids(X, K);
endif

idx = findClosestCentroids(X, initial_centroids);

centroids = computeCentroids(X, idx, K);

max_iters = 10;

[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);

fprintf('Program paused. Press enter to continue.\n');
end
