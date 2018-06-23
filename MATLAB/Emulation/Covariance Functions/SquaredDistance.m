function C = SquaredDistance(A, B, l)
% Computes a matrix of all pairwise squared distances between two sets of
% vectors taking in to account modifications to length scale.
%
% INPUT
% A - (n x d) - Matrix of d column vectors of length n.
% B - (m x d) - Matrix of d column vectors of length m.
%             - If non-existant or empty, taken to be A.
% l - (1 x d) - Row vector of length d.
%             - Modifies the length scale of the columns of A and B.
%             - If non-existant or empty, taken to be ones.
%
% OUTPUT
% C - (n x m) - Pairwise square distances between the two vectors
%                taking the modifications to length scale into account.

%% Variable Checks

if ~exist('A','var') || isempty(A)
    error('Matrix A doesn''t exist or is empty.');
end

dimsA = size(A);
if nargin == 1 || isempty(B)
    B = A;
    dimsB = dimsA;
else
    dimsB = size(B);
    if dimsA(2) ~= dimsB(2)
        error('A and B must have the same number of columns.');
    end
end

%% Calculation

if ~exist('l','var') || isempty(l)
    l = ones(dimsA(2),1);
end

C = zeros(dimsA(1),dimsB(1));

for d = 1:dimsA(2)
    a_matrix = repmat(A(:,d)./l(d), 1, dimsB(1));
    b_matrix = repmat(B(:,d)'./l(d), dimsA(1), 1);
    C = C + ((a_matrix - b_matrix).^2);
end

end