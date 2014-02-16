function plotRegression(X,Y,f,f_name)

figure;
plot(X,Y,'o',X,f,'-','color','r');
legend('Training Data',f_name);
xlabel('time since impact(ms)');
ylabel('force on head (g)');
title(f_name);
end
