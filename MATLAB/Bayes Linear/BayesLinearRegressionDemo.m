% Bayes Linear Regression - Demo Script
%% Model
% I consider 2 models, both where x is considered a non-random quantity:
% Model 1: Yi = Alpha + xi * Beta + Epsi
% Model 2: {Yi = Alpha + xi * Beta, Zi = Yi + Epsi}
% Both of these are valid models. Which one we select depends on the problem we are doing.
% If there was no observational error, but Yi wasn't always the same value for a given xi, Alpha and
% Beta, then Model 1 would make more sense.
% If we consider Yi to be completely determined by xi, Alpha and Beta but we could only gain insight into it
% through measurements contaminated with observational error, then Model 2 would make more sense.

%% Model 1
x = -1:0.1:3;
x = x'; % Want column vector.
Alpha = -3;
Beta = 2;
Eps = 0.25.*randn(size(x,1),1); % N(0,0.25^2);
Y = Alpha + x*Beta + Eps;

figure;
plot(x, Alpha + x .* Beta , 'k-')
hold on
xlabel('x')
ylabel('Y')
grid on
legend('Y = -3 + 2x', 'Location', 'best')

%% Priors
E_Alpha = 0; % It's useful to also see what we learn we use the true values for Alpha and Beta as our prior.
E_Beta = 0; % (It turns out using [0, 0] gives a very similar result to using the true values which is good).
V_Alpha = 3^2; % 3^2
V_Beta = 1^2; % 1^2
Cov_Alpha_Beta = 0;

E_Epsi = 0;
V_Epsi = 0.25^2; % I assume I know the variance of Epsi.
% If we do not know V_Epsi, the simplest thing to do is use the least squares estimate for V_Epsi as a point
% estimate.
% I also assume Eps is uncorrelated with all other quantities.

% Consequentially,
% E[Yi] = E[Alpha + xi * Beta + Epsi]
%       = E[Alpha] + E[xi * Beta] + E[Epsi]
%       = E[Alpha] + xi * E[Beta] + 0
E_Y = E_Alpha + x .* E_Beta;

% Var[Yi] = Var[Alpha + xi * Beta + Epsi]
%         = Var[Alpha] + Var[xi * Beta] + Var[Epsi]    since all covariances are 0.
%         = Var[Alpha] + xi^2 * Var[Beta] + Var[Epsi]
V_Y =  V_Alpha + (x.^2 * V_Beta) + V_Epsi;

% Cov[Yi, Yj] = Cov[Alpha + xi * Beta + Epsi, Alpha + xj * Beta + Epsj]
%             = Var[Alpha] + xi * xj * Var[Beta]   since all covariances are 0.
VC_Y = V_Alpha + ((x*x') .* V_Beta) + V_Epsi .* eye(size(Y,1)); % Diagonals are V_Y.

% Cov[Alpha, Yi] = Cov[Alpha, Alpha + xi * Beta + Epsi]
%                = Var[Alpha]    since all covariances are 0.
C_Alpha_Y = V_Alpha .* ones(1, size(Y,1));

% Cov[Beta, Yi] = Cov[Beta, Alpha + xi * Beta + Epsi]
%               = Cov[Beta, xi * Beta]    since all covariances are 0.
%               = xi * Var[Beta]
C_Beta_Y = V_Beta .* x'; 

% In total:
% D = [Y1; Y2; ... Yn];
D = Y;
E_D = E_Y;
VC_D = VC_Y; % Prior variance-covariance matrix of collection D.

% B = [Alpha; Beta];
% Note: We learn nothing about Epsi.
% This is because it is apriori uncorrelated with all other quantities.
% I therefore do not include it in B.
E_B = [E_Alpha; E_Beta];
VC_B = [V_Alpha, Cov_Alpha_Beta,; ...
    Cov_Alpha_Beta', V_Beta];

C_B_D = [C_Alpha_Y; C_Beta_Y]; % Covariance matrix (non-square in general) for covariance between collections B and D.

plot(x, E_Alpha + x .* E_Beta , 'g-')
plot(x, E_Alpha + x .* E_Beta + 2.*sqrt(V_Y) , 'g--')
plot(x, E_Alpha + x .* E_Beta - 2.*sqrt(V_Y) , 'g--', 'HandleVisibility', 'off')

plot(x, E_Alpha + (sqrt(V_Alpha) * randn) + (x .* (sqrt(V_Beta) * randn)), '-', 'Color' , [0,0.5,0])
legend('Y = -3 + 2x', 'Prior Mean Line' , 'Prior +-2 SDs', 'Prior Sample Line', 'Location', 'best')

%% Data
plot(x, Y, 'k+')
legend('Y = -3 + 2x', 'Prior Mean Line' , 'Prior +-2 SDs', 'Prior Sample Line', 'Observed', 'Location', 'best')

%% Bayes Linear Adjustments
E_D_B = E_B + C_B_D * pinv(VC_D) * (D - E_D); % Adjusted Expectation

VC_D_B = VC_B - C_B_D * pinv(VC_D) * C_B_D'; % Adjusted Variance

% Var_D[Yi] = Var_D[Alpha + xi * Beta + Epsi]
%           = Var_D[Alpha] + Var_D[xi * Beta] + 2Cov_D[Alpha, xi * Beta] + Var_D[Epsi]
%           = VC_D_B(1,1) + xi^2 * VC_D_B(2,2) + 2xi * Cov_D[Alpha, Beta] + Var[Epsi]
%           = VC_D_B(1,1) + xi^2 * VC_D_B(2,2) + 2xi * VC_D_B(1,2) + Var[Epsi]

Var_D_Y = VC_D_B(1,1) + (x.^2 .* VC_D_B(2,2)) + (2 .* x .* VC_D_B(1,2));
Var_D_YObs = Var_D_Y + V_Epsi; % The variance for an obseration.

plot(x, E_D_B(1) + x .* E_D_B(2), 'b-')
plot(x, E_D_B(1) + x .* E_D_B(2) + 2.*sqrt(Var_D_Y) , 'b--')
plot(x, E_D_B(1) + x .* E_D_B(2) - 2.*sqrt(Var_D_Y) , 'b--', 'HandleVisibility', 'off')
legend('Y = -3 + 2x', 'Prior Mean Line' , 'Prior +-2 SDs', 'Prior Sample Line', 'Observed', 'Adjusted Mean Line', 'Adjusted +-2 SDs', 'Location', 'best')
