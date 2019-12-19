function [BetaHat, Xbar, SigmaSqEst, VarBetaHat, RSq] = LeastSquaresSolveQR(X, F, Y)
% A function to find the least squares estimates (MLE assuming homoscedastic Gaussian errors) of
% regression coefficients. A QR decomposition is used for stability.
% NOTE: To predict at new test points X_star, Xbar must be first subtract from X_star:
%  e.g. Y_star = F(X_star - Xbar) * BetaHat;
%       Var_Y_star = diag(F(X_star - Xbar) * VarBetaHat * F(X_star - Xbar)');
%
% Inputs:
% X = Covariates
%   - Dimension (n x q) where n is the number of training points and q is the number of covariates. 
% F = Design matrix function
%   - (anonymous) Function.
%   - Example: F = @(X)[ones(size(X, 1), 1), X(:,1), X(:,1).^2, X(:,2), X(:,2).^2, X(:,1).*X(:,2), exp(X(:,2))];
% Y = Training points.
%   - Dimension (n x m) where n is the number of training points and m is the number of outputs.
%
% Outputs:
% E_Beta = Least sqares estimates of the regression coefficients. 
%        - Dimension (p x m) where p is the number of fitted parameters and m is the number of outputs.
%        - mdl.Coefficients.Estimate from fitlm.
% Xbar = Mean of covariates.
%      - Dimension (1 x q).
% SigmaSqEst = Standard error of the residuals squared
%            - Dimension (m x m) where m is the number of outputs.
%            - mdl.MSE from fitlm
% V_Beta = Variance-Covariance matrix of the least squares estimates.
%        - Dimension (mp * mp)
%        - mdl.CoefficientCovariance from fitlm
% RSq = Coefficient of Determination.
%     - Dimension scalar.
%     - mdl.Rsquared.Adjusted from fitlm.

%% Centring and Design Matrix
% Centre X by its mean so predictors will be orthohonal to the intercept.
Xbar = mean(X);
X = X - Xbar;
D = F(X); % Design matrix.

%% Checks
[n, p] = size(D);

% Intercept Check: check if the design matrix has a column of ones (corresponding to an intercept coefficient) in
% it's first column:
if ~isequal(D(:,1), ones(n, 1))
    error('LeastSquaresSolveQR expects the first column of the design matrix to be ones.')
end

% Dimension checks:
if n <= p
    if n == p
        warning('Number of parameters to be fit is equal to the number of data points.')
    else
        error('Can only fit at most as many parameters as data points.')
    end
end

%% QR Decomposition
[Q, R] = qr(D,0); % economy-size / reduced QR decomposition. Only returns the first p columns of Q and rows of R.
% Note that q(11,2) is very small compared to the rest of q. I expect it should be 0 but isn't due
% to rounding.

Q(:,1) = sign(R(1,1)) .* ones(n,1) ./ sqrt(n);

%% Estimation
BetaHat = R \ (Q' * Y);
% Let design matrix X = Q*R.

% Understanding 1:
% Residuals r = Y - X * Beta
%             = Y - Q * R * Beta
% Left multiply by Q'
%      Q' * r = Q' * Y - (Q' * Q) * R * Beta
%             = Q' * Y - R * Beta = u
% r' * r = r' * Q' * Q * r as Q is orthonormal.
% So, r' * r = u' * u and is minimised when u is zero.
% This occurs when R * BetaHat = Q' * Y
% So, BetaHat = R^-1 * Q' * Y

% Understanding 2:
% BetaHat = (X' * X)^-1 * X' * Y
%         = ((Q * R)' * Q * R)^-1 * (Q * R)' * Y
%         = (R' * Q' * Q * R)^-1 * R' * Q' * Y
%         = (R' * R)^-1 * R' * Q' * Y    as Q'*Q = I as Q is orthonormal.
%         = R^-1 * R'^-1 * R' * Q' * Y
%         = R^-1 * Q' * Y


SigmaSqEst = (1/(n-p)) * (Y - (Q*R*BetaHat))'*(Y - (Q*R*BetaHat)); % NOTE: size(X , 1) - size(X , 2) = n - p
% (Using the notation that p = number of parameters)
% The Variance Matrix for the residuals (Standard Error of the Residuals Sq). 
% If Mulitple Output, this will be a dimension (p x p) matrix where SigmaSqEst(i,j) = Cov(E_i , E_j) where
% E_k is the vector of residuals associated with the kth output.

VarBetaHat = kron( SigmaSqEst , (R \ (R' \ eye(p))) ); % kron takes each element in A and .*s it by B.
% This is a dimension (mp x mp) matrix. It is m^2 blocks of size (p x p), each of which is
% VarCov(Beta^i , Beta^j) where Beta^k is the vector of coefficients for output k.
% i.e. the top left most corner (p x p) block is the VarCov matrix of the Beta associated with the
% first output. 
% If we fit straight lines to two outputs a and b, the full VarCov matrix would be:
% VarCov(B^a , B^a) VarCov(B^a , B^b)
% VarCov(B^b , B^a) VarCov(B^b , B^b)
% and looking closer at the top 2x2 sub-matrix, we would see:
% Cov(B_0^a , B_0^b) Cov(B_0^a , B_1^b)
% Cov(B_1^a , B_0^b) Cov(B_1^a , B_1^b)

RSq =  (1- SigmaSqEst./var(Y));

end