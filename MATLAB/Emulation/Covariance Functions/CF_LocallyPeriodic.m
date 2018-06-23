function K = CF_LocallyPeriodic( X , Xstar , L_p , p , L)
% Takes in column vectors.

% To get a periodicity in a single direction, use something similar to the following:
% KXX1L_Single = CF_LocallyPeriodic( X , X , 1 , [17, inf] , [17*4 , 17*4]);


K = CF_Periodic( X , Xstar , L_p , p) .* CF_SquaredExponential( X , Xstar , L );

end