
close all; clc

setSize = 10 % number of points per generated cluster

K = 10;

if ~exist('X', 'var')
% generate a random set X separated by 10's
    X = [];

    for i = 0:(K-1)
        X = cat(1, X, rand(setSize,2) + (i * 10));
        i = i + 1;
    end
endif

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
