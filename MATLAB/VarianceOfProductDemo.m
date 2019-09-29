clc
nSamples = 1e6;
offsetX = 1;
offsetY = 1;
X = offsetX + randn(nSamples, 1); % When offset = 0, ~N(0,1) --> mu_X = 0, sigmaSq_X = 1;
Y = offsetY + rand(nSamples, 1); % When offset = 0, ~U(0,1) --> mu_Y = 0.5, sigmaSq_Y = 1/12;
mu_X = 0 + offsetX;
mu_Y = 0.5 + offsetY;
sigmaSq_X = 1;
sigmaSq_Y = 1/12;
disp(['X ~ N(', num2str(mu_X), ',1) --> mu_X = ', num2str(mu_X), ', sigmaSq_X = 1.']);
disp(['Y ~ U(', num2str(offsetY + 0), ',', num2str(offsetY + 1), ') --> mu_Y = ', num2str(mu_Y), ', sigmaSq_Y = 1/12.']);

disp(['Take ', num2str(nSamples), ' samples of X and Y.']);
Xbar = mean(X);
sSq_X = var(X);
disp(['Xbar = ', num2str(Xbar), ', sSq_X = ', num2str(sSq_X), '.']);
Ybar = mean(Y);
sSq_Y = var(Y);
disp(['Ybar = ', num2str(Ybar), ', sSq_Y = ', num2str(sSq_Y), '.']);

disp('Let Z = XY.')
Z = X.*Y;
mu_Z = mu_X * mu_Y;
disp(['mu_Z = mu_X * mu_Y = ', num2str(mu_X * mu_Y), ' as X and Y are independent.'])

Zbar = mean(Z);
sSq_Z = var(Z);
disp(['Zbar = ', num2str(Zbar), ', sSq_Z = ', num2str(sSq_Z), '.']);

%% Using Formula Var(XY) = Var(X)Var(Y) + Var(X)[E(Y)]^2 + Var(Y)[E(X)]^2
% Holds when X and Y are independent.
sigmaSq_Z_Stats = sigmaSq_X * sigmaSq_Y + sigmaSq_X * (mu_Y^2) + sigmaSq_Y * (mu_X^2);

%% Using a "Propogation of Errors" Formula
% https://faraday.physics.utoronto.ca/PVB/Harrison/ErrorAnalysis/Propagation.html
% "Everything is this section assumes that the error is "small" compared to the value itself, i.e.
% that the fractional error is much less than one."

% Again, for when X and Y are independent.
sigmaSq_Z_Physics = mu_Z^2 * ( (sigmaSq_X / mu_X^2) + (sigmaSq_Y / mu_Y^2) );
% Produces NaN since we divide by mu_X = 0.

% Rewriting this, we get
% sigmaSq_Z_Physics = mu_X^2 * mu_Y^2 * ( (sigmaSq_X / mu_X^2) + (sigmaSq_Y / mu_Y^2) );
%                   = sigmaSq_X * (mu_Y)^2 +  sigmaSq_Y * (mu_X^2)

% Because of the "small error assumption", they remove the Var(X)Var(Y) term because it would be
% small^2. If they had not done this, the formula would be equivalent.