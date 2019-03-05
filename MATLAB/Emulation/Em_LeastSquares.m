function [ BetaHat , SigmaSq , VarBetaHat , RSq ] = Em_LeastSquares( X , Y )
% A function to find the least squares estimates of regression coefficients.
% A cholesky decomposition is used for stable inversion of X'*X.

% Inputs 
% X = Design Matrix - Dimension (n x p) where n is the number of training points and p is the number of
% fitted parameters (in simple linear regression, p = 2 since we have Beta_0 and Beta_1).
% Y = Training points - Dimension (n x m) where n is the number of training training points and m is
% the number of outputs.

% Outputs
% E_Beta = Least sqares estimates of the regression coefficients - Dimension (p x m) where p is the 
% number of fitted parameters and m is the number of outputs.
% SigmaSq = Standard error of the residuals squared - Dimension (m x m) where m is the number of
% outputs.
% V_Beta = Variance-Covariance matrix of the least squares estimates - Dimension (mp * mp) 
% RSq = Coefficient of Determination - Dimension scalar.

L = chol( X'*X , 'lower' );
BetaHat = L' \ (L \ (X'*Y) ); % 
% A = LL' = U'U, % Cholesky Decomposition
% A^-1 = (L^-1)'(L^-1) = (U^-1)(U^-1)' % Cholesky Inversion
% But note that for an invertible square matrix Z, (Z^-1)' = (Z')^-1, so
% A^-1 = (L')^-1(L^-1) = (U^-1)(U')^-1
% Proof: (Z^-1)' = (Z^-1)'Z'(Z')^-1 % Multiplying by Identiy Matrix, I
%                = (ZZ^-1)'(Z')^-1  % Using A'B' = (BA)'
%                = I'(Z')^-1
%                = (Z')^-1

SigmaSq = (1/(size(X , 1) - size(X , 2))) * (Y - X*BetaHat)'*(Y - X*BetaHat); % NOTE: size(X , 1) - size(X , 2) = n - p
% (Using the notation that p = number of parameters)
% The Variance Matrix for the residuals (Standard Error of the Residuals Sq). 
% If Mulitple Output, this will be a dimension (p x p) matrix where SigmaSq(i,j) = Cov(E_i , E_j) where
% E_k is the vector of residuals associated with the kth output.

VarBetaHat = kron( SigmaSq ,   (L' \ (L\ eye(size(X , 2)))) ); % kron takes each element in A and .*s it by B.
% This is a dimension (mp x mp) matrix. It is m^2 blocks of size (p x p), each of which is
% VarCov(Beta^i , Beta^j) where Beta^k is the vector of coefficients for output k.
% i.e. the top left most corner (p x p) block is the VarCov matrix of the Beta associated with the
% first output. 
% If we fit straight lines to two outputs a and b, the full VarCov would be:
% VarCov(B^a , B^a) VarCov(B^a , B^b)
% VarCov(B^b , B^a) VarCov(B^b , B^b)
% and looking closer at the top right entry, we would see:
% Cov(B_0^a , B_0^b) Cov(B_0^a , B_1^b)
% Cov(B_1^a , B_0^b) Cov(B_1^a , B_1^b)

RSq =  (1- SigmaSq./var(Y));

end % End of function