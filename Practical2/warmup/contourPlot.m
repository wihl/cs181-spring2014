%Generates a colour filled contour plot of the bivariate function, 'func'
%over the domain [-1,1]x[-1,1], plotting it to the current figure. Also plots
%the specified point as a white cross. 


function contourPlot(func,trueValue)
stepSize = 0.05; 
[x,y] = meshgrid(-1:stepSize:1,-1:stepSize:1); % Create grid.
[r,c]=size(x);

data = [x(:) y(:)];
p = func(data);
p = reshape(p, r, c);
 
contourf(x,y,p,256,'LineColor','none');
colormap(jet(256));
axis square;
set(gca,'XTick',[-1,0,1]);
set(gca,'YTick',[-1,0,1]);
xlabel(' W0 ');
ylabel(' W1 ','Rotation',0);
if(length(trueValue) == 2)
    hold on;
    plot(trueValue(1),trueValue(2),'+w');
end
end
