function a__L_CF = NN_CF_CrossEntropy( a__L , y )
% Neural Network Cost Function
% a__L and y are n x 1 vectors.

a__L_CF = -sum( ( y.*ln(a__L) ) + ( (1-y).*ln(1-a) ) );

end