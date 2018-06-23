function [a__l , z__l] = NN_FF( x , W , B , z , AF)
% Neural Network Feed Forward
% Function takes a single training sample x (1 x p), a weight matrix stack W (cell array), a bias vector stack B (cell array),
% a weighting function z and an activation function AF.
% Function produces outputs a__L (the activations of the neurons in the output (final) layer of a Neural
% Network) and all of the weighted sums z__l for all neurons in all layers.

% W and B have to be cell arrays since we do not require the same number of neurons in each layer.

z__l = cell(1 , length(B));
a__l = cell(1 , length(B));

for c = 1:length(B);
    if c == 1
        %         a = sigma(x , W{c} , B{c});
        z__l{1,c} = z(x , W{c} , B{c});
        a__l{1,c} = AF(z__l{1,c});
        continue
    end
    %     a = sigma(a , W{c} , B{c});
    z__l{1,c} = z(a__l{1,c-1} , W{c} , B{c});
    a__l{1,c} = AF(z__l{1,c});
end

end