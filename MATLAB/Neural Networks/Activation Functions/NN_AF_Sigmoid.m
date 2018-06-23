function a__l = NN_AF_Sigmoid( z__l )
% Neural Network Activation Function
% a__lm1 and b__l are ? x 1 vectors and w__l is a ? x ? matrix

a__l = 1./(1 + exp(-z__l));
end