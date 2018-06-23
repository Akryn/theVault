%% Specify Clouds in a 2D space

mu1 = [3 ; -5];
mu2 = [0 ; 12];
mu3 = [-7 ; 3];

sigma_X1 = 1;
sigmasq_X1 = sigma_X1^2;
sigma_X2 = 2;
sigmasq_X2 = sigma_X2^2;
sigma_X3 = 2;
sigmasq_X3 = sigma_X3^2;

sigma_Y1 = 2;
sigmasq_Y1 = sigma_Y1^2;
sigma_Y2 = 3;
sigmasq_Y2 = sigma_Y2^2;
sigma_Y3 = 3;
sigmasq_Y3 = sigma_Y3^2;

% It's not always easy to specify a covariance. What is easy to specify is a correlation.
cor_XY1 = 0.7; % Recall correlation p = cov(X,Y) / (sigma_X * sigma_Y).
cov_XY1 = cor_XY1 * sigma_X1 * sigma_Y1;
cor_XY2 = 0;
cov_XY2 = cor_XY2 * sigma_X2 * sigma_Y2;
cor_XY3 = -0.7;
cov_XY3 = cor_XY3 * sigma_X3 * sigma_Y3;

Sigma1 = [sigmasq_X1 , cov_XY1; ...
    cov_XY1 , sigmasq_Y1];
Sigma2 = [sigmasq_X2 , cov_XY2; ...
    cov_XY2 , sigmasq_Y2];
Sigma3 = [sigmasq_X3 , cov_XY3; ...
    cov_XY3 , sigmasq_Y3];


%% Samples (Used as a visualisation aid)

samp1 = randn(2,10000);
z1 = [(chol(Sigma1 , 'Lower') * samp1) + mu1]';

samp2 = randn(2,10000);
z2 = [(chol(Sigma2 , 'Lower') * samp2) + mu2]';

samp3 = randn(2,10000);
z3 = [(chol(Sigma3 , 'Lower') * samp3) + mu3]';

t = [0 ; 0];

%%

plot(z1(:,1) , z1(:,2) , 'r.' , 'MarkerSize' , 1)
hold on
plot(mu1(1) , mu1(2) , 'kx')

plot(z2(:,1) , z2(:,2) , 'b.' , 'MarkerSize' , 1)
plot(mu2(1) , mu2(2) , 'kx')

plot(z3(:,1) , z3(:,2) , 'g.' , 'MarkerSize' , 1)
plot(mu3(1) , mu3(2) , 'kx')

plot(t(1) , t(2) , 'k*')

grid on
xlabel('x')
ylabel('y')

plot([mu1(1) , t(1)] , [mu1(2) , t(2)] , 'k--')
plot([mu2(1) , t(1)] , [mu2(2) , t(2)] , 'k--')
plot([mu3(1) , t(1)] , [mu3(2) , t(2)] , 'k--')
axis image

%% Euclidean Distance

ed1 = sqrt( (t-mu1)' * (t-mu1) );
ed2 = sqrt( (t-mu2)' * (t-mu2) );
ed3 = sqrt( (t-mu3)' * (t-mu3) );

%% Mahalanobis Distance

md1 = sqrt(  ((t-mu1)' / Sigma1) * (t-mu1));
md2 = sqrt(  ((t-mu2)' / Sigma2) * (t-mu2));
md3 = sqrt(  ((t-mu3)' / Sigma3) * (t-mu3));

%%

disp(['Euclidean distances for [R , B , G] are [' , num2str(ed1) , ' , ' , num2str(ed2) , ' , ' , num2str(ed3) , ']'])
disp(['Mahalanobis distances for [R , B , G] are [' , num2str(md1) , ' , ' , num2str(md2) , ' , ' , num2str(md3) , ']'])


