function [Q, R, orthonormPredict] = Rpoly(x, degree, intercept)
% Function to create orthonormal polynomial design matrix. 
% Only supports a single covariate. 
% All monomials are included up to and including the specified degree with the potential exception
% of the constant term.
%
% Adapted from
% https://stackoverflow.com/questions/39031172/how-poly-generates-orthogonal-polynomials-how-to-understand-the-coefs-ret/39051154#39051154
% https://stats.stackexchange.com/questions/253123/what-are-multivariate-orthogonal-polynomials-as-computed-in-r
% is also a good resource.
%
% Inputs: 
% x (n x 1): Covariate.
% degree (1 x 1): The requested degree of the polynomial.
% intercept {0,1}: Flag to include the constant term. Default = 1.
%
% Outputs: 
% Q (n x {degree, degree + 1}): Orthonormal polynomial design matrix, potentially without
%   the constant term.
% orthonormPredict (struct): Structure used in Rpoly_predict.

%% Intercept Check
if ~exist('intercept', 'var') || isempty(intercept)
    intercept = true;
end

%% Centring and Design Matrix
% Centre x by its mean so covariates will be orthohonal to the intercept.
xbar = mean(x);
x = x - xbar;

powers = 0:degree;
X = x.^powers;

%% Dimension Checks

if size(X,1) <= size(X,2)
    if size(X,1) == size(X,2)
        warning('Number of parameters to be fit is equal to the number of data points.')
    else
        error('Can only fit at most as many parameters as data points.')
    end
end

%% QR Decomposition

[Q, R] = qr(X,0); % economy-size / reduced QR decomposition. Only returns the first p columns of Q and rows of R.
% Note that q(11,2) is very small compared to the rest of q. I expect it should be 0 but isn't due
% to rounding.

Q(:,1) = sign(R(1,1)) .* ones(size(x,1),1) ./ sqrt(size(x,1));

%% Prediction Structure
if nargout > 1
    % Recall that the diagonal elements of R are <q_i, v_i> = ||w_i||, i.e. the L2-Norm (the square
    % root of the sum of squares) of the ith unnormalised orthogonal basis vector.
    
    % Because MATLAB's qr() does not always return diagonal elements of R as positives, ||w_i|| can
    % be calculated as negative.
    scale = diag(R)';
    W = Q .* scale;
    
    alpha = (sum(x .* (W.^2)) ./ (scale.^2)); % my_poly uses 2:end-1
    beta = (scale.^2) ./ [size(X,1), (scale(1:end-1).^2)]; % my_poly uses 2:end-1
    
    orthonormPredict.degree = degree;
    orthonormPredict.xbar = xbar;
    orthonormPredict.scale = scale;
    orthonormPredict.alpha = alpha;
    orthonormPredict.beta = beta;
    orthonormPredict.intercept = intercept;
end

%% Intercept
if ~intercept
    Q = Q(:,2:end);
end

end % End of function.