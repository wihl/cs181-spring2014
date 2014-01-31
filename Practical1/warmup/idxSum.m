function s = idxSum(idx, K)

s = zeros(K,1);

for i = 1:K
    s(i) = sum(idx == i);
end

end
