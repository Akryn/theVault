function [G_xstar , Var_fstar] = GPWithMarginalisedBetandSigma(X,Xstar,Y,l,Sigman,thi)

% X is training inputs - input as column vector
% Y is traing outputs -input as column vector
% Xstar is input values for prediction - input as column vector
% l is the correlation length coefficient WARNING very touchy.
% Sigman is the "Nugget", our noise on the training outputs which we have observed. 
% thi is a function to create a design matrix.

%Note: Throughout the comment walkthrough we use (A^-1)' = (A')^-1 for square matrices.
%Proof: (A')(A^-1)' = (A^-1 * A)'    because A' * B' = (B * A)' 
%                   = I'
%                   = I


H = thi(X); %Design Matrix for training data.

Hstar = thi(Xstar); %Design Matrix for test data.

KXX = CF_SquaredExponential(X,X,l); %Covariance Matrix of the training data WITHOUT noise.

U = chol((KXX + (Sigman^2)*(eye(size(X,1))))); %Cholesky decposition (Upper Traiangular matrix) of the variance matrix (covariance & noise).
%Cholesky Inversion: A^-1 = (U^-1) * (U^-1)' where U is lower triangular matrix satisfying A = U'U
%NOTE: Ultimately, we will use: (U)^-1 (U^-1)' = { K(X,X) + (Sigman^2)I }^-1

alpha = U\(U'\Y); %Equivalent to (U)^-1 (U')^-1 * Y    =  (U)^-1 (U^-1)' * Y  by properties of matricies
%                                                      =  { K(X,X) + (Sigman^2)I }^-1 * Y      
%                              - this is what we multiply the vector K(X,Xstar) to get our predicted values in simple GPs.

dimXstar = size(Xstar); 
KXstarX = CF_SquaredExponential(Xstar,X,l); % K(X,Xstar) - covariances of the training data to each test data point (matrix).
G_xstar = zeros(dimXstar(1),1); %Initialising final predictions.
Var_fstar = zeros(dimXstar(1),1); %Initialising final variances.

for i = 1 : dimXstar(1) 
    G_xstar(i) =  KXstarX(i,:) * alpha;  %The simple GP prediction as described above in alpha: K(X,Xstar) * { K(X,X) + (Sigman^2)I }^-1 * Y 
    v = U'\KXstarX(i,:)'; %Equivalent to (U')^-1 * K(Xstar,X) = (U^-1)' * K(Xstar,X)
    kXstarXstar = CF_SquaredExponential(Xstar(i,:),Xstar(i,:),l); %Covariance Matrix of the test data WITHOUT noise.
    Var_fstar(i) = (kXstarXstar.^2 + Sigman.^2) - v'*v; % The simple GP variance (WITH noise) as v'*v = K(X,Xstar)' *(U^-1)(U^-1)' * K(Xstar,X)
                                                        %                                            = K(X,Xstar)' *{ K(X,X) + (Sigman^2)I }^-1 * K(Xstar,X)  
                                                        % So Var_fstar(i) = [ K(Xstar,Xstar) + Sigman^2 ] - [ K(X,Xstar)' *{ K(X,X) + (Sigman^2)I }^-1 * K(Xstar,X) ]
end

v = U'\H; %Equivalent to (U')^-1 * H = (U^-1)' * H

BetaHat = v'*v \ (( H'*alpha )); % = {H' * (U^-1)(U^-1)' * H}^-1 * H' * {U\(U'\Y)}
                                 % = {H' * [ K(X,X) + (Sigman^2)I) ]^-1 * H}^-1 * H' * { K(X,X) + (Sigman^2)I }^-1 * Y 
                                 % Which is the Toolkit estimate for Beta (Also the MLE for Beta in
                                 % Generalised Least Squares and REML for Beta).
                                 
u = U'\(Y-H*BetaHat); % = (U')^-1 * (Y-H*BetaHat) = (U^-1)' * (Y-H*BetaHat)

ScaledSigmafHatsq = u'*u; % = [ (U')^-1 * (Y-H*BetaHat) ]' * [ (U^-1)' * (Y-H*BetaHat) ]
                    % = [ (Y-H*BetaHat)' * { ( (U')^-1)' } ] * [ (U^-1)' * (Y-H*BetaHat) ]
                    % = [ (Y-H*BetaHat)' * { ( (U')')^-1 } ] * [ (U^-1)' * (Y-H*BetaHat) ]
                    % = (Y-H*BetaHat)' * [ (U)^-1 * (U^-1)' ] * (Y-H*BetaHat)
                    % = (Y-H*BetaHat)' * [ K(X,X) + (Sigman^2)I ]^-1 * (Y-H*BetaHat)
                    % Which is (n-q-2) times the Toolkit Estimate for Sigmaf^2 where q is the number of
                    % predictors in our Least Squares.

R = (Hstar - (KXstarX*(U\v))); % = Hstar - ( K(Xstar,X) * [ (U^-1) * { (U')^-1 * H } ] ) 
                               % = Hstar - ( K(Xstar,X) * [ { (U^-1) * (U^-1)' } * H ] )
                               % = Hstar - ( K(Xstar,X) * [ K(X,X) + (Sigman^2)I ]^-1 * H ] )

G_xstar = (G_xstar + R*BetaHat); % = [ K(X,Xstar) * { K(X,X) + (Sigman^2)I }^-1 * Y ] + Hstar * BetaHat - ( K(Xstar,X) * [ K(X,X) + (Sigman^2)I ]^-1 * H ] ) * BetaHat
                                 % = Hstar * BetaHat + [ K(X,Xstar) * { K(X,X) + (Sigman^2)I }^-1 ] * {Y - (H * BetaHat)}
                                 % Which is the GP prediction with a non-zero mean function which
                                 % here is a regression with weak prior.

Var_fstar = (Var_fstar +  bsxfun_method(R/(v'*v),R')); %sum(bsxfun(@times, A , B.') , 2 ) = bsxfun_method prevents the creation of large matrices saving memeory.

%         = [ K(Xstar,Xstar) + Sigman^2 ] - [ K(X,Xstar)' *{ K(X,X) + (Sigman^2)I }^-1 * K(Xstar,X) + sum( [ R / (v' * v) ] .* R , 2) ]
% Where [ R / (v' * v) ] .* R does array element wise multiplication and then the sum sums over the columns giving a column vector.
% This is equivalent to selecting the diagonal elements of [ { Hstar - ( K(Xstar,X) * [ K(X,X) + (Sigman^2)I ]^-1 * H ] )} * {H' * [ K(X,X) + (Sigman^2)I) ]^-1 * H}^-1 ] * { Hstar - ( K(Xstar,X) * [ K(X,X) + (Sigman^2)I ]^-1 * H ] ) } ]
% i.e. the diagonal elements of R/(v'*v)*R' which are the variances from the covariance matrix.

% The full Covariance Matrix would be = [ K(Xstar,Xstar) + Sigman^2 ] - K(X,Xstar)' *{ K(X,X) + (Sigman^2)I }^-1 * K(Xstar,X) + [ { Hstar - ( K(Xstar,X) * [ K(X,X) + (Sigman^2)I ]^-1 * H ] )} * {H' * [ K(X,X) + (Sigman^2)I) ]^-1 * H}^-1 ] * { Hstar - ( K(Xstar,X) * [ K(X,X) + (Sigman^2)I ]^-1 * H ] ) } ] 

Var_fstar = (ScaledSigmafHatsq/(length(Y)-size(H,2)-2)) .* Var_fstar; % Mutliplying variance by scaling factor sigma after rescaling it by 1/(n-q-2) as 1/(n-q) relates to a GP, but this is actually a TP (t (distribution) process)
                                                                      % and the variance of a t dist. is (n-q)/(n-q-2) times that of a normal dist.

if (isnan(Var_fstar))
    warning('Variance NaN');
end

if (isnan(G_xstar))
warning('Variance NaN')
end

end