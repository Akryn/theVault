nSamp = 1e8;
maxDim = 40;
rEdges = 0:0.01:10;


for dim = 1:maxDim
    
    X = randn(dim, nSamp);
    if dim > 1
        r = sqrt(sum(X.^2));
    else
        r = X;
    end
    rDiscretize = discretize(r, rEdges);
    
    rCounts = zeros(length(rEdges),1);
    for a = 1 : length(rEdges)
        rCounts(a) = sum(rDiscretize == a);
    end
    
    rProbMass = rCounts ./ nSamp;
    plot(rEdges + 0.05, rProbMass, 'DisplayName', num2str(dim));
    if dim == 1
        hold on
        legend('Location', 'best')
        xlabel('r')
        ylabel('Prob. Mass')
    end
end