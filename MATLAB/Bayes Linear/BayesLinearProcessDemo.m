% Bayes Linear Residual Process - Demo Script
%% Model
% 1).
% We have Y = f(x) which we would like to learn about. f is an unknown deterministic function. 
% Therefore, we treat Y as random since we have uncertainty in the function f.
% We can learn about it through noisy observations z = f(x) + eps(x)
% We assume eps has expectation 0 and is uncorrelated with Y.

% 2).
% We have Y = f(x) which we would like to learn about. f is an unknown stochastic function.
% Then Y is random since we have uncertainty in the function f, and f is random.
% We then typically partition uncertainty in Y into uncertainy f and the random nature of f.
% We can learn about it through observations yi = f(xi)

% I will assume Model 1:
% Y = f(x)
% Z = f(x) + Eps(x)
% E[Eps(x)] = 0
% V[Eps(x)] = sigma_nSq
% Cov[Esp(x1), Eps(x2)] = 0 for x1 ~= x2
% E[Y] = 0
% VarCov[Y] = covFun(x1, x2) 
% E[Z] = E[Y] + E[Eps(x)] = 0 + 0 = 0
% VarCov[Z] = Cov[f(x1) + Eps(x1), f(x2) + Eps(x2)] 
%           = Cov[f(x1), f(x2)] + Cov[f(x1), Eps(x2)] + Cov[Eps(x1), f(x2)] + Cov[Eps(x1), Eps(x2)]
%           = covFun(x1, x2) + sigma_nSq * delta(x1, x2) 
% Cov[Y,Eps(x)] = 0
% Cov[Y,Z] = Cov[Y, Y + Eps(x)] = Cov[Y,Y] + Cov[Y,Eps(x)] = Cov[Y,Y] + 0 = Var[Y]

% As Eps(x) is uncorrelated with everything (except itself), we will not resolve any uncertainty
% about it.

%% Initialisation
x = -4*pi : pi/4 : 4*pi; % The points we see z at. Default -4*pi : pi/4 : 4*pi
x = x'; % Want column vector.
xstar = -6*pi : pi/64 : 6*pi; % The points we are interesting in observing Y at (includes x).
xstar = xstar';

sigma_nSq = 0.025^2; % The observational uncertainty.
Y = sin(xstar); % The underlying true value of Y at xstar, unknown to us.

figure;
plot(xstar, Y, 'k-')
hold on
xlabel('x')
ylabel('Y')
grid on
legend('Y = sin(x)', 'Location', 'best')

%% Priors
E_D = zeros(size(x)); 
E_B = zeros(size(xstar));
% Too cumbersome to specify my prior covariance beliefs individually for all points so will adpot a
% covariance function.
% Simple covariance functions take the form sigmaSq * f(d(x1,x2))
dist = @(x1,x2)(bsxfun(@minus, x1, x2'));

l = 1; % Correlation Length
covFunExp = @(x1,x2)(exp(-abs(dist(x1,x2)) / l));
covFunSqExp = @(x1,x2)(exp( - (dist(x1,x2).^2) ./ (2*l^2) ));

p = 2*pi; % Period
lp = 1; % Rate of Decay of Periodic Correlation
covFunPeriodic = @(x1,x2)(cos(2*pi*dist(x1,x2)/p));
covFunPeriodicExp = @(x1,x2)(exp( (-2*sin(pi*dist(x1,x2)/p).^2) / lp^2 ));
covFunLocalPeriodicExp = @(x1,x2)(exp( (-2*sin(pi*dist(x1,x2)/p).^2) / lp^2 ) .* exp( - (dist(x1,x2).^2) ./ (2*l^2) ));

% figure;
% plot(xstar, covFunExp(0, xstar))
% hold on
% plot(xstar, covFunSqExp(0, xstar))
% plot(xstar, covFunPeriodic(0, xstar))
% plot(xstar, covFunPeriodicExp(0, xstar))
% plot(xstar, covFunLocalPeriodicExp(0, xstar))
% xlabel('$dist(x_i,x_j)$', 'Interpreter', 'LaTeX')
% xticks([-6*pi -5*pi -4*pi -3*pi -2*pi -pi 0 pi 2*pi 3*pi 4*pi 5*pi 6*pi])
% xticklabels({'-6\pi', '-5\pi', '-4\pi', '-3\pi','-2\pi','-\pi','0','\pi','2\pi','3\pi', '4\pi', '5\pi', '6\pi'})
% set(gca,'xaxisLocation','top')
% ylabel('Correlation', 'Interpreter', 'LaTeX')
% leg1 = legend('Exponential, $l = 2\pi$' , ...
%     'Squared Exponential, $l = 2\pi$' , ...
%     'Periodic, $p = 2\pi$' , ...
%     'Periodic Exponential, $p = 2\pi$, $l_p = 1$' , ...
%     'Locally Periodic Exponential, $p = 2\pi$, $l_p = 1$, $l = 2\pi$',...
%     'Location', 'best');
% set(leg1,'Interpreter','LaTeX');
% clear leg1

covFun = covFunPeriodic; % Selecting from the covariance functions

% VC_D = covFun(x,x) + sigma_nSq .* eye(size(x,1)); % sigma_nSq is taken to be known.
VC_D = covFun(x,x) + sigma_nSq .* eye(size(x,1)) + 1e-6 .* eye(size(x,1)); % With nugget
% VC_B = covFun(xstar,xstar);
VC_B = covFun(xstar,xstar) + 1e-6 .* eye(size(xstar,1));% With nugget
% Errors occurredin MATLAB when sampling the prior distribution when we tried to calculate a
% Cholesky decomposition of VC_B stating the matrix was not positive definite.
% The addition of a "nugget" term, a small constant added to the diagonals of the VarCov
% matrices ensures numerical stability. The value is small in comparison to the size of sigma_nSq.

% Cov[Y,Z] = Cov[Y, Y + eps] = Cov[Y, Y] + Cov[Y, eps] = Cov[Y, Y] = Var[Y]
C_B_D = covFun(xstar,x);

plot(xstar, E_B, 'g-')
plot(xstar, E_B + 2.*sqrt(diag(VC_B)) , 'g--')
plot(xstar, E_B - 2.*sqrt(diag(VC_B)) , 'g--', 'HandleVisibility', 'off')

plot(xstar, E_B + chol(VC_B, 'lower') * randn(size(xstar,1),1), '-', 'Color' , [0,0.5,0])
legend('Y = sin(x)', 'Prior Mean Line' , 'Prior +-2 SDs', 'Prior Sample Line', 'Location', 'best')

%% Data
z = sin(x) + sqrt(sigma_nSq) * randn(size(x,1),1); % N(0,0.025^2);
d = z;
plot(x, z, 'k+')
legend('Y = sin(x)', 'Prior Mean Line' , 'Prior +-2 SDs', 'Prior Sample Line', 'Observed', 'Location', 'best')

%% Bayes Linear Adjustments
E_D_B = E_B + C_B_D * pinv(VC_D) * (d - E_D); % Adjusted Expectation

VC_D_B = VC_B - C_B_D * pinv(VC_D) * C_B_D'; % Adjusted Variance

plot(xstar, E_D_B, 'b-')
plot(xstar, E_D_B + 2.*sqrt(diag(VC_D_B)) , 'b--')
plot(xstar, E_D_B - 2.*sqrt(diag(VC_D_B)) , 'b--', 'HandleVisibility', 'off')
legend('Y = sin(x)', 'Prior Mean Line' , 'Prior +-2 SDs', 'Prior Sample Line', 'Observed', 'Adjusted Mean Line', 'Adjusted +-2 SDs', 'Location', 'best')

disp(['SD at 0 is ', num2str(sqrt(VC_D_B(xstar == 0,xstar == 0)))])
% Note:
% covFun = covFunExp has SD at 0 of 0.025028, slightly above 0.025
% covFun = covFunSqExp has SD at 0 of 0.0247, slightly below 0.025
% covFun = covFunPeriodic has SD at 0 of 0.00615, far below 0.025

% Why does using Exp and SqExp covariance functions cause the remaining uncertainty in Y at the
% training points to be very close to the sigma_nSq for my example?
% This phenomenon does not occurr when we scale the SqExp covariance function by a small constant, e.g.
% 0.0000001, however, my priors no longer make sense as all the training points not very close to 0
% are apriori highly unlikely.
% The phenomenon also does not occurr when using SqExp if add more training points creating,
% creating small gaps between each, e.g. -4*pi : pi/64 : 4*pi instead of my original -4*pi : pi/4 : 4*pi
% This change does not invalidate my priors as far as I can tell.
% Could it be I just happened to set up my priors and training points to hit a certain sweet
% spot, and in general we do not get the posterior variance at a point to be the observational
% variance?

% No, kind of. We are inside the large region of prior and data combinations where this can occurr.
% If we instead chose our training points when using SqExp to be -4*pi : pi : 4*pi, then the
% training points are spaced far enough away from each other so that a training point is highly
% correlated with itself and uncorrelated with all other training points (note that a correlation
% length can change this). Then, the posterior variance on that training point is the variance of
% the observational noise.
% As we obtain finer spaced training sets, fixing everything else, we will obtain training points
% with strong correlation between each other meaning we can reduce our posterior uncertainty at
% training points below the observational variance. Points which are extremely correlated
% effectively behave like extra observations for each other, effectively allowing us to average over
% the noisy observations to get the mean value and use the standard error of the mean as the
% uncertainty. This can best be seen when we use the Periodiccovariance function where training 
% points are said to be almost perfectly correlated (the observational noise we add to training
% data prevents them being perfectly correlated).