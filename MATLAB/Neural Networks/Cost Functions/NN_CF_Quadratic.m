function a__L_CF = NN_CF_Quadratic( a__L , y )
% Neural Network Cost Function
% a__L and y are n x 1 vectors.

a__L_CF = 0.5*sum( (a__L - y).^2);

end