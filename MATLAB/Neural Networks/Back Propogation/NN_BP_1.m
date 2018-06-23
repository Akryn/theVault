function [d__L] = NN_BP_1( grad_a__L_CF , AF_Prime_L )
% Calculates error in last layer.

d__L = grad_a__L_CF .* AF_Prime_L;

end