function [z__l] = NN_AF_z__l( a__lm1 , w__l , b__l )
% Neural Network Activation Function
% a__lm1 and b__l are ? x 1 vectors and w__l is a ? x ? matrix

z__l = a__lm1*w__l + b__l;
end