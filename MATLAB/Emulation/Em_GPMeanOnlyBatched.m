function m = Em_GPMeanOnlyBatched(X, Xstar, Y, l, Sigman, DesignMatrixFunction, CorrFun)
% Gaussian process emulator with marginalised Beta.
% Does not calculate variance.
% Detectos how much available physical memory exists to avoid swapping.

if( ~exist( 'CorrFun','var'))
    CorrFun = @CF_SquaredExponential;
end

% [X, xNormFactor] = normalizeMatrixEntries(X);
% [Xstar, xStarNormFactor] = normalizeMatrixEntries(Xstar);
% l = l/xNormFactor;

%% Problem Checks
% sys.PhysicalMemory.Available is amount of memory (in bytes) shown as "Available" on Task Managers'
% Memory page on the Performance tab. It does not include swap/page file space.
[~ , sys] = memory;
% Assuming we only work with doubles in MATLAB, the maximum number of elements we can store in our workspace
% and not swap is sys.PhysicalMemory.Available / 8 since doubles are 64 bits which is 8 bytes.
maxNumElements = sys.PhysicalMemory.Available / 8;

% How much memory would be use if we had KXX and L in our workspace at the same time?
nX = size(X,1);
numElementsForXandL = 2*(nX^2);

swapIndicator = ceil( numElementsForXandL / maxNumElements );

if swapIndicator > 1
    XSkip = ceil(sqrt(swapIndicator));
    warning(['Training points alone cause us to swap memory. Training points will be sampled using every ' , num2str(XSkip) , 'th value.'])
    X = X(1:XSkip:end,:);
    nX = size(X,1);
    Sigman = Sigman(1:XSkip:end);
    Y = Y(1:XSkip:end,:);
end

%% Shared by all Batches
D = DesignMatrixFunction(X);
KXX = CorrFun( X , X , l );
if length(Sigman) == 1
    Sigman = repmat(Sigman, nX , 1);
end
L = chol( KXX + diag(Sigman) , 'lower');    clear KXX;

if( sum(D(:,1) , 1) == 0 )
    BetaHat = 0;
    alpha = L \ (Y - D*BetaHat);
else
    w = L \ D;
    Q = w' *w;  clear w;
    K = chol(Q , 'lower');  clear Q;
    BetaHat = K' \ (K \ D') * (L' \(L \ Y)) ; % GLS estimate for Beta
    alpha = L \ (Y - D*BetaHat);
end

clearvars -except Xstar X l thi BetaHat CorrFun L alpha

%% Figuring out how to Batch

% sys.PhysicalMemory.Available is amount of memory (in bytes) shown as "Available" on Task Managers'
% Memory page on the Performance tab. It does not include swap/page file space.
[~ , sys] = memory;
% Assuming we only work with doubles in MATLAB, the maximum number of elements we can store in our workspace
% and not swap is sys.PhysicalMemory.Available / 8 since doubles are 64 bits which is 8 bytes.
maxNumElements = sys.PhysicalMemory.Available / 8;

% How much memory would be use if tried to calculate the output for all Xstar at once?
nXstar = size(Xstar,1);
numElementsForAllXstar = nXstar^2;

nSplits = ceil( numElementsForAllXstar / maxNumElements );
splitSize = floor( nXstar / nSplits ) ;

%% Batch

m = zeros(nXstar , 1);

if nSplits > 1
    h=waitbar(0,'Initialising','Name','SRM_GPEmulatorSigmafInputMeanOnlyBatched');
end

for a = 1:nSplits
    
    if nSplits > 1
        waitbar(a/nSplits,h,['Batch ' , num2str(a) , ' of ' , num2str(nSplits)]);
    end
    
    thisSplit = [((a-1)*splitSize)+1 ,  a*splitSize];
    Hstar = DesignMatrixFunction( Xstar(thisSplit(1) : thisSplit(2) , :) );
    KXXstar = CorrFun( X , Xstar(thisSplit(1) : thisSplit(2) , :) , l );
    m(thisSplit(1) : thisSplit(2)) = (Hstar * BetaHat) + KXXstar' * (L' \ alpha);
end

if nSplits > 1
    close(h)
end

end % End of function