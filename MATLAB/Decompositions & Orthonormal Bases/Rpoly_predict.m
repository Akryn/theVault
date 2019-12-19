function Qstar = Rpoly_predict(xstar, orthonormPredict)
% Function to create design matrix for prediction which will correspond with a training design
% matrix generated with Rpoly.
% Only supports a single covariate. 
% Note that the generated design matrix will not in general by orthonormal.
%
% Adapted from
% https://stackoverflow.com/questions/39031172/how-poly-generates-orthogonal-polynomials-how-to-understand-the-coefs-ret/39051154#39051154
% https://stats.stackexchange.com/questions/253123/what-are-multivariate-orthogonal-polynomials-as-computed-in-r
% is also a good resource.
%
% Inputs: 
% xstar (n x 1): Covariate.
% orthonormPredict (struct): Structure generated from Rpoly.
%
% Outputs: 
% Qstar (n x {orthonormPredict.degree, orthonormPredict.degree + 1}): Design matrix, potentially
%   without the constant term.

%% Qnpack Prediction Structure
degree = orthonormPredict.degree;
xbar = orthonormPredict.xbar;
scale = orthonormPredict.scale;
alpha = orthonormPredict.alpha;
beta = orthonormPredict.beta;
intercept = orthonormPredict.intercept;

%% Centring
xstar = xstar - xbar;

%% Generating New Test Points
Xstar = repmat(xstar, 1, degree);

if degree > 1
    Xstar(:,2) = (xstar - alpha(2)) .* Xstar(:,1) - beta(2);
    
    if degree > 2
        a = 3;
        while a <= degree
            Xstar(:,a) = (xstar - alpha(a)) .* Xstar(:,a-1) - beta(a) .* Xstar(:,a-2);
            a = a + 1;
        end
    end
    
end

Xstar = [ones(size(Xstar,1),1), Xstar];
Qstar = Xstar ./ scale;

if ~intercept
    Qstar = Qstar(:,2:end);
end

end % End of function