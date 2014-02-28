
% Fix the random seed for repeatability.
rng(1);

cifar_raw = load('~/Data/CIFAR-100/train.mat');
fprintf('CIFAR-100 Data Loaded\n');

% Cast the unsigned bytes into doubles and standardize.
X = standardize(double(cifar_raw.data));

% Set number of clusters.
K = 16;

% tic/toc times things. This takes about 265 seconds on my MacBook Pro.
tic;
[R mu] = kmeans(X, K, 1);
toc

% Make the means easier to visualize.
mu = mu - min(mu(:));
mu = mu / max(mu(:));

% Montage prints images, the permute/reshape just munges the
% dimensions so that montage knows what to do with them.
figure(1);
montage(permute(reshape(mu', [32 32 3 K]), [2 1 3 4]));

figure(2);
for kk=1:K
  Xk = cifar_raw.data(R==kk,:);
  Xk = Xk(1:25,:);

  subplot(4,4,kk);
  montage(permute(reshape(Xk', [32 32 3 size(Xk,1)]), [2 1 3 4]));
	
end