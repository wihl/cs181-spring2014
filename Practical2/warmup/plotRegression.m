function plotRegression(X,Y,f,f_name)

figure;
hold on
plot(X,Y,'or',X,f,'-m','linewidth',3);
errorbar(X,Y,f,"b");

legend('Training Data',f_name);
xlabel('time since impact(ms)');
ylabel('force on head (g)');
title(f_name);
end
