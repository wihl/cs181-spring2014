function plotSampleLines(mu, sigma, numberOfLines,dataPoints)
% Plot the specified number of lines of the form y = w0 + w1*x in [-1,1]x[-1,1] by
% drawing w0, w1 from a bivariate normal distribution with specified values
% for mu = mean and sigma = covariance Matrix. Also plot the data points as
% blue circles. 
for i = 1:numberOfLines
    model = struct('mu', mu, 'Sigma', sigma);
    w = gaussSample(model);
    func = @(x) w(1) + w(2)*x;
    fplot(func,[0,60],'r');
    hold on;
end
axis square;
%set(gca,'XTick',[-1,0,1]);
%set(gca,'YTick',[-1,0,1]);
%xlabel(' x ');
%ylabel(' y ','Rotation',0);
if(size(dataPoints,2) == 2)
    hold on;
    plot(dataPoints(:,1),dataPoints(:,2),'ob');    
end
end
