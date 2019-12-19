% SVD (https://en.wikipedia.org/wiki/Singular_value_decomposition) is the generalisation of the eigendecomposition
% (https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix).
% I will only consider the case where A is real.
%
% A = U * Sigma * V' where:
% A = m x n matrix,
% U = m x m orthonormal matrix,
% Sigma = m x n rectangular diagonal matrix with non-negative real numbers on the diagonal,
%                and these diagonal entries sigma_i of Sigma are the singular values (sigma_i =
%                sqrt(lambda_i) where lambda_i is an eigenvalue of A),
% V = n x n orthonormal matrix.

% A' * A = (V * Sigma' * U') * (U * Sigma * V')
%        = V * Sigma' * Sigma * V'
% This is a diagonalisation (eigendecomposition) of A' * A.

% A * A' = (U * Sigma * V') * (V * Sigma' * U')
%        = U * Sigma * Sigma' * U'
% This is a diagonalisation (eigendecomposition) of A * A'.

% A * V = U * Sigma
% because V^-1 = V' since V is orthonormal.

%% Implementing SVD from Example
% Similar to https://www.youtube.com/watch?v=cOUTpqlX-Xs
% This IS NOT a good algorithm to use in practice, it merely demonstrates theoretically how the
% decomposition can be performed.

A = [5, 5;
    -1, 7];

% First compute A' * A
AtA = A' * A;
% [26, 18
%  18, 74];

% Find eigenvalues and eigenvectors of A' * A
[V,Lambda] = eig(AtA);

% Recall that Sigma = sqrt(Lambda)
Sigma = sqrt(Lambda);

% Calculate A * V
AV = A * V;

% Calculate A * V * Sigma^-1 = U
% Recall that Sigma^-1 = 1 ./ Sigma since Sigma is diagonal.

U = AV * diag((1 ./ diag(Sigma)));

%% Using MATLAB's SVD Function

[Umat, Sigmamat ,Vmat] = svd(A);
