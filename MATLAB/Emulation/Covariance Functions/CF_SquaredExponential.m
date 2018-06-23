function K = CF_SquaredExponential( X , Xstar , L )
% Takes in column vectors.

K = SquaredDistance(X,Xstar,L);
K = exp(-(K)/2);
    
end