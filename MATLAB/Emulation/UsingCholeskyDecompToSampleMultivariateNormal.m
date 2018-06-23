% Specify mean vector and VarCov matrix.

mu_X = 1;
mu_Y = 2;

mu = [mu_X ; mu_Y];

sigma_X = 1;
sigmasq_X = sigma_X^2;

sigma_Y = 2;
sigmasq_Y = sigma_Y^2;

% It's not always easy to specify a covariance. What is easy to specify is a correlation.
cor_XY = 0.7; % Recall correlation p = cov(X,Y) / (sigma_X * sigma_Y).
cov_XY = cor_XY * sigma_X * sigma_Y;

Sigma = [sigmasq_X , cov_XY; ...
    cov_XY , sigmasq_Y];

%% Sample - 1D

% Given we can sample from the standard Normal (Gaussian) distribution (N(0,1)), we sample from any
% Gaussian distribution as follows:

% To transform to the standard Normal, subtract the mean and multiply by the sqrt of the
% variance:
%
% z = (x - mu) / sigma_X
%
% Inverting this, we see:
% (sigma_X * z) + mu_X = x

x = sigma_X * randn(1000000,1) + mu_X;
histogram(x)

%% Sample - 2D

% The approach to sampling in 2D is very similar. The only thought that needs to be done is on what
% the sqrt of the VarCov matrix Sigma is.

% Such a matrix is of the form L satisfying LL' = Sigma or U satisfying UU' = Sigma. Such matrices
% can be found via the Cholesky decomposition.
% https://makarandtapaswi.wordpress.com/2011/07/08/cholesky-decomposition-for-matrix-inversion/

samp = randn(2,1000000);

z_Ue = [(chol(Sigma) * samp) + mu]';

z_Le = [(chol(Sigma , 'Lower') * samp) + mu]';

z_eU = (samp'*chol(Sigma)) + mu';

z_eL = (samp'*chol(Sigma,'Lower')) + mu';

figure;

subplot(2,2,1)
plot(z_Ue(:,1) , z_Ue(:,2) , '.')
title('Ue - Wrong')
xlim([-10 , 10])
ylim([-10 , 10])

subplot(2,2,2)
plot(z_Le(:,1) , z_Le(:,2) , '.')
title('Le - Right')
xlim([-10 , 10])
ylim([-10 , 10])

subplot(2,2,3)
plot(z_eU(:,1) , z_eU(:,2) , '.')
title('eU - Right')
xlim([-10 , 10])
ylim([-10 , 10])

subplot(2,2,4)
plot(z_eL(:,1) , z_eL(:,2) , '.')
title('eL - Wrong')
xlim([-10 , 10])
ylim([-10 , 10])

% We want a sample which has 0 mean and VarCov Sigma.
% e is a vector of elements sampled from N(0,1).
% Using a Cholesky Decomposition, we can find L such that LL^T = Sigma.
% So Var[Le] = L Var[e] L^T = Sigma since Var(e) = I (or 1 when 1 dim).

% A similar approach can show we correctly generate samples when using eU,
% assuming Var(eU) = U^T Var(e) U
