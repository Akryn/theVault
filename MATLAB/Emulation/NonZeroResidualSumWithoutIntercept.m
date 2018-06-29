x = [0:0.1:10]';
y = 2.*x + randn(size(x));
plot(x,y , '.')

%% No Intercept

b = SRM_LeastSquaresSolveChol( x , y );

yhat = x*b;
e = yhat - y;

disp(sum(e));

%% Intercept

D = [ones(size(x)) , x];
ab = SRM_LeastSquaresSolveChol( D , y );

yhat = D*ab;
e = yhat - y;

disp(sum(e));

