function K = CF_Periodic( X , Xstar , L_p , p)
% Takes in column vectors.

K = sqrt(SquaredDistance(X,Xstar,p));

K = exp( -(2.*( sin(pi.*K).^2) ) ./ L_p^2);   
end