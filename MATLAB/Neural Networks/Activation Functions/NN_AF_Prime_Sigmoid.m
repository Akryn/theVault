function [AF_Prime_l] = NN_AF_Prime_Sigmoid(z__l)
% Neural Network Derivative of Activation Function w.r.t. z evaluated at z__l.

AF_Prime_l = NN_AF_Sigmoid(z__l) .* (1 - NN_AF_Sigmoid(z__l));

end