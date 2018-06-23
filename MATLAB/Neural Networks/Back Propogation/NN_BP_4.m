function [dCbydw_jk_l] = NN_BP_4( d_j_l , a_k_lm1)
% Neural Network Activation Function
% a__lm1 and b__l are ? x 1 vectors and w__l is a ? x ? matrix

dCbydw_jk_l = a_k_lm1 * d_j_l;

end