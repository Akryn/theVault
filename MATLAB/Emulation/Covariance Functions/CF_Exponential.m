function K = CF_Exponential( X , Xstar , L )
% Takes in column vectors.

K = exp( -0.5*( sqrt( SquaredDistance(X , Xstar , L) )) );


end