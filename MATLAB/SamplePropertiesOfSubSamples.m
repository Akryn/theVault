nSamp = 1000;
samp = randn(nSamp);

mu = 0;
sigmaSq = 1^2;

E_Xbar = mu;
Var_Xbar = sigmaSq / nSamp;

xbar = mean(samp(:));
sSq = var(samp(:));

nPerm = 100;
xbar_sub = zeros(nSamp, nPerm);
sSq_sub = zeros(nSamp, nPerm);

% Generate subsamples of sample, find mean and variance of all subsamples, regenerate subsamples and repeat.
for b = 1 : nPerm
    samp = reshape(samp(randperm(numel(samp))), nSamp, nSamp);
    
for a = 1 : nSamp
xbar_sub(a,b) = mean(samp(a,:));
sSq_sub(a,b)= var(samp(a,:));
end
disp(b)
end

%%
figure;

% Mean of subsamples = Mean of sample (independent of partition that generates subsamples)
% However, it is still random and does not equal mu in general.
xbar_subFull = mean(xbar_sub); 
% All identical to compuational error
subplot(2,2,1)
plot(xbar_subFull, 'r')
title('Mean of Subsample Means for Permutations of Sample')
xlabel('Permutation of Sample')

% Variance of subsample means depends on the partition that generates subsamples.
% The mean of this set has expected value Var_Xbar = sigmaSq / nSamp.
sxbar_sub = var(xbar_sub);
subplot(2,2,2)
plot(sxbar_sub, 'r.')
hold on
plot([1, nPerm], [Var_Xbar, Var_Xbar], 'b')
title('Variance of Subsample Means for Permutations of Sample')
xlabel('Permutation of Sample')

% Mean of subsample variances depends on the partition that generates subsamples.
% If we were to repeat this entire experiment many times, we would see the mean of this set would be
% on average sigmaSq.
sSq_subbar = mean(sSq_sub);
subplot(2,2,3)
plot(sSq_subbar, 'r.')
title('Mean of Subsample Variances for Permutations of Sample')
xlabel('Permutation of Sample')

% The sum of the Mean of Subsample Variances and Variance of Subsample Means seems to be independent
% of the partititon that generates the subsamples.
% However, it is still random.
% If we were to repeat this entire experiment many times, we would see the mean of this set would be
% on average sigmaSq(1 + 1/nSamp).
sSq_subFull = sxbar_sub + sSq_subbar;
% All identical to compuational error
subplot(2,2,4)
plot(sSq_subFull, 'r')
title('(Variance of Subsample Means) + (Mean of Subsample Variances) for Permutations of Sample')
xlabel('Permutation of Sample')