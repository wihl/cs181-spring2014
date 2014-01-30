
clear; close all; clc

K = 5;
setSize = 10;
X = [];

% generate a random set X separated by 10
for i = 0:(K-1)
  X = cat(1, X, rand(setSize,2) + (i * 10));
  i = i + 1;
end

%initial_centroids = [ 5 5; 15 15; 28 28; 38 38; 42 42];
initial_centroids = kMeansInitCentroids(X, K);

idx = findClosestCentroids(X, initial_centroids);

centroids = computeCentroids(X, idx, K);

max_iters = 10;

[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);

fprintf('Program paused. Press enter to continue.\n');
pause;
