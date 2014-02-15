input = csvread('motorcycle.csv' );
X = input(:,1);
Y = input(:,2);
xlabel('time since impact (ms)')
ylabel('g-force on head')
p = polyfit(X,Y,9);
f = polyval(p,X);
plot(X,Y,'o',X,f,'-')
