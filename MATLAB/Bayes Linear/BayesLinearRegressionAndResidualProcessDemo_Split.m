% Bayes Linear Regression & Residual Process - Demo Script
% See BayesLinearRegressionAndResidualProcess_DemoCombined First.
%% Model
% We would like to learn about Y = g(x).
% Knowledge of Y informs us that it can be thought of as being having a linear global trend but with
% higher frequency local variations from the trend.
% We use the following model:
% Y = Alpha + x * Beta + f(x)
% where the Alpha + x * Beta represent the linear global trend
% and f(x) represents the higher frequency local variations from the trend.
% We assume the linear global trend and the local variations are uncorrelated.
% Now, we would like to learn about Alpha, Beta and f(x) instead of Y as a wgole.
% We can learn about them through noisy observations Z = Y + Eps.
% We assume Eps has expectation 0 and is uncorrelated with all quantities. 

%% Initialisation
x = -4*pi : pi/4 : 4*pi;
x = x'; % Want column vector.
xstar = -6*pi : pi/64 : 6*pi; % The points we are interesting in observing Y at (includes x).
xstar = xstar';

Alpha = 0.5;
Beta = 0.1;

sigma_nSq = 0.025^2; % The observational uncertainty. Taken to be known. = Var[Eps]
fxstar = sin(xstar); % The underlying true value of Y at xstar, unknown to us.

Y = Alpha + xstar .* Beta + fxstar; 

figure;
plot(xstar, Y, 'k-')
hold on
xlabel('x')
ylabel('Y')
grid on
legend('Y', 'Location', 'best')

%% Priors
E_Alpha = 0; % It's useful to also see what we learn we use the true values for Alpha and Beta as our prior.
E_Beta = 0; % (It turns out using [0, 0] gives a very similar result to using the true values which is good).
V_Alpha = 0.5^2; %  0.5^2;
V_Beta = 0.2^2; % 0.2^2;
Cov_Alpha_Beta = 0;

E_fX = zeros(size(x));
E_fXstar = zeros(size(xstar));

% Too cumbersome to specify my prior covariance beliefs individually for all points so will adpot a
% covariance function.
% Simple covariance functions take the form sigmaSq * f(d(x1,x2))
dist = @(x1,x2)(bsxfun(@minus, x1, x2'));
covFunExp = @(x1,x2)(exp(-abs(dist(x1,x2))));
covFunSqExp = @(x1,x2)(exp(-(dist(x1,x2).^2)./2));
covFunPeriodic = @(x1,x2)(cos(dist(x1,x2)));
% For more stuff on covariance functions, see BayesLinearProcessDemo

covFun = covFunPeriodic; % Selecting from the covariance functions

VC_fX = covFun(x,x) + 1e-6 .* eye(size(x,1)); 
VC_fXstar = covFun(xstar,xstar) + 1e-6 .* eye(size(xstar,1));
% The addition of a "nugget" term, a small constant added to the diagonals of the VarCov
% matrices ensures numerical stability. The value is small in comparison to the size of sigma_nSq.

E_Eps = 0;
V_Eps = sigma_nSq;

% Cov[fX, fXstar] = Cov[Y, Y + eps] = Cov[Y, Y] + Cov[Y, eps] = Cov[Y, Y] = Var[Y]
C_fXstar_fX = covFun(xstar,x);

% E[Y] = E[Alpha + x * Beta + f(x)]
E_Y = zeros(size(xstar));
% E[Z] = E[Alpha + x * Beta + f(x) + Eps]
E_Z = zeros(size(x));

% Cov[Y] = Cov[Alpha + x1 * Beta + f(x1), Alpha + x2 * Beta + f(x2)]
%        = Cov[Alpha, Alpha] + x2*Cov[Alpha, Beta] + Cov[Alpha, f(x2)]
%          + x1*Cov[Beta, Alpha] + x1*x2*Cov[Beta, Beta] + x1*Cov[Beta, f(x2)]
%          + Cov[f(x1), Alpha] + x2*Cov[f(x1), Beta] + Cov[f(x1), f(x2)]
%        = Var[Alpha] + x1*x2*Var[Beta, Beta] + Var[f(x1), f(x2)]
VC_Y = V_Alpha + xstar*xstar'.*V_Beta + VC_fXstar;

VC_Z = V_Alpha + x*x'.*V_Beta + VC_fX + V_Eps .* eye(size(x,1));

plot(xstar, E_Y, 'g-')
plot(xstar, E_Y + 2.*sqrt(diag(VC_Y)) , 'g--')
plot(xstar, E_Y - 2.*sqrt(diag(VC_Y)) , 'g--', 'HandleVisibility', 'off')

% Originally, I incorrectly calculated VC_Y missing off the x1*x2 infront of Var[Beta].
% I therefore sampled correctly when I sampled Alpha, Beta and f separately but not when sampling Y.
% The following two lines are both valid ways of sampling (now that the calculation of VC_Y is correct above).
% Note that they will not give the same sample since one samples one matrix and the other samples 2
% scalars and a matrix.
% allAtOnce = E_Y + chol(VC_Y, 'lower') * randn(size(xstar,1),1);
% byParts = E_Y + (sqrt(V_Alpha) * randn) + (xstar .* (sqrt(V_Beta) * randn)) + (chol(VC_fXstar, 'lower') * randn(size(xstar,1),1));

plot(xstar, E_Y + chol(VC_Y, 'lower') * randn(size(xstar,1),1), '-', 'Color' , [0,0.5,0])
legend('Y', 'Prior Mean Line' , 'Prior +-2 SDs', 'Prior Sample Line', 'Location', 'best')

% B = [Alpha; Beta; f(xstar)]
% D = [Z]

E_B = [E_Alpha; E_Beta; E_fXstar];
E_D = E_Z;

% Cov[Alpha, f(xstar)] = 0;
% Cov[Beta, f(xstar)] = 0;

VC_B = [V_Alpha, 0, zeros(1, size(E_fXstar,1));...
    0, V_Beta, zeros(1, size(E_fXstar,1));...
    zeros(size(E_fXstar,1), 2) , VC_fXstar];
VC_D = VC_Z;

% Cov[Alpha, Z] = Cov[Alpha, Alpha + x * Beta + f(x) + Eps]
%               = Var[Alpha]
% Cov[Beta, Z] = Cov[Beta, Alpha + x * Beta + f(x) + Eps]
%              = xVar[Beta]
% Cov[f(xstar), Z] = Cov[f(xstar), Alpha + x * Beta + f(x) + Eps]
%                  = Cov[f(xstar), f(x)]

C_B_D = [V_Alpha .* ones(1, size(E_fX,1)); ...
    x' .* V_Beta .* ones(1, size(E_fX,1)); ...
    C_fXstar_fX];

%% Data
fx = sin(x); % The underlying true value of Y at xstar, unknown to us.
eps = randn(size(x,1),1);
z = Alpha + x .* Beta + fx + sqrt(sigma_nSq) * eps; % N(0,0.025^2);
d = z;
plot(x, z, 'k+')
legend('Y', 'Prior Mean Line' , 'Prior +-2 SDs', 'Prior Sample Line', 'Observed', 'Location', 'best')

%% Bayes Linear Adjustments
E_d_B = E_B + C_B_D * pinv(VC_D) * (d - E_D); % Adjusted Expectation

VC_d_B = VC_B - C_B_D * pinv(VC_D) * C_B_D'; % Adjusted Variance
% (2,1) and (1,2) are not the same value (Cov[Alpha,Beta]), meaning this isn't a valid variance matrix.
% Caused by numerical instability. Didn't observe this in the _Combined version when we don't
% consider the regression parameters by themselves.

E_d_Y = E_d_B(1) + xstar.* E_d_B(2) + E_d_B(3:end);

% Var_d[Y] = Var_d[Alpha + xstar .* Beta + f(xstar)]
%          = Var_d[Alpha] + xstar.^2 .* Var_d[Beta] + Var_d[f(xstar)] + xstar .* 2Cov[Alpha, Beta] + 2Cov[Alpha,
%          f(xstar)] + xstar .* 2Cov[Beta, f(xstar)]
V_d_Y = VC_d_B(1,1) + (xstar .^2) .* VC_d_B(2,2) + diag(VC_d_B(3:end,3:end)) + xstar .* 2.*VC_d_B(1,2) +...
    2.*VC_d_B(3:end,1) + xstar .* 2.*VC_d_B(3:end,2); 
% Matches with diag(VC_z_Y) from BayesLinearRegressionAndResidualProcess_DemoCombined

plot(xstar, E_d_Y, 'b-')
plot(xstar, E_d_Y + 2.*sqrt(diag(V_d_Y)) , 'b--')
plot(xstar, E_d_Y - 2.*sqrt(diag(V_d_Y)) , 'b--', 'HandleVisibility', 'off')
legend('Y', 'Prior Mean Line' , 'Prior +-2 SDs', 'Prior Sample Line', 'Observed', 'Adjusted Mean Line', 'Adjusted +-2 SDs', 'Location', 'best')

disp(['SD at 0 is ', num2str(sqrt(V_d_Y(xstar == 0)))])
