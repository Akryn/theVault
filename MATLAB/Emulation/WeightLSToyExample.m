F = @(x)([ones(size(x,1),1) , x]);
x = [[0:10]' ; 2 ; 2 ; 2 ; 2];
n = length(x);

e = 0.1.*randn(n,1);

y = 2.*x + 1 + e; 

plot(x,y , 'ko')
hold on


D = F(x);
Beta = WeightedLS(D , diag( ones(n,1) ) ./ n , y);
plot(x,D*Beta, 'bx-')

%%
x2 = [0:10]';
y2 = y;
y2(3) = mean([y(3) ; y(12:end)]);
y2(12:end) = [];
D2 = F(x2);
W3 = ones(length(x2) , 1);
W3(3) = 5;
W3 = W3 ./ length(W3);
Beta2 = WeightedLS(D2 , diag(W3) , y2);
plot(x2,D2*Beta2, 'gx-')

% Same Beta as before.

%%

x2 = [0:10]';
y3 = y2;
y2(3) = mean([y(3) ; y(12:end)]);
y2(12:end) = [];
D2 = F(x2);
W3 = ones(length(x2) , 1);
W3 = W3 ./ length(W3);
Beta3 = WeightedLS(D2 , diag(W3) , y2);
plot(x2,D2*Beta3, 'rx-')

% Difference Beta as before.