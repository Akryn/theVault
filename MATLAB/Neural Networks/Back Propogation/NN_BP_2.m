function [d__l] = NN_BP_2( d__lp1 , AF_Prime_l , w__lp1 )
% Neural Network Activation Function
% a__lm1 and b__l are ? x 1 vectors and w__l is a ? x ? matrix

d__l = (d__lp1*(w__lp1')) .* AF_Prime_l;

end