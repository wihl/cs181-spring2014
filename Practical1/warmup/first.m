
K = 5;
setSize = 10;
X = [];

% generate a random set X separated by 10
for i = 0:(K-1)
  X = cat(1, X, rand(setSize,2) + (i * 10));
  i = i + 1;
end




