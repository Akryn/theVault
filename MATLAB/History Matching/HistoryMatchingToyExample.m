clc
close all

f = @(x) 3.*sin(x) + x - (x.^2)/4 + exp(x/4); % Function is deterministic, no noise.
x = [0:0.01:10]';
x = round(x,2); % required to match later applications of round to random numbers.
y = f(x);

figure;
plot(x,y , 'k');
xlabel('x')
ylabel('y')
title('1st Iteration')
hold on

xTest = 5; % Unknown when history matching. This is what we want to find out.
yTest = f(xTest);
plot(xTest, yTest, 'kx')

%% Sample

nSamp = 3;
xSamp1 = ((max(x) - min(x)).*lhsdesign(nSamp,1)) + min(x);
xSamp1 = sort(xSamp1);
xSamp1 = round(xSamp1,2);
% [~,indSamp1] = ismember(xSamp1, x);
% ySamp1 = y(indSamp1);
ySamp1 = f(xSamp1);

plot(xSamp1, ySamp1 , 'bx')

%% GP
% From Rasmussen Algorithm 2.1

% Prior N(0, CF_SquaredExponential)
l = 1;
sigmaSq_f = 10;
sigmaSq_n = 1e-6; % Typically noise, but in this case just for numerical stability.
KxSamp1xSamp1 = sigmaSq_f .* CF_SquaredExponential(xSamp1, xSamp1, l) + eye(length(xSamp1)) .* sigmaSq_n;
L = chol(KxSamp1xSamp1 , 'lower');
alpha = L'\(L\ySamp1);

KxSamp1x = sigmaSq_f .* CF_SquaredExponential(xSamp1, x, l);
mFuncPostSamp1Atx = KxSamp1x' * alpha; % Posterior mean function @ x after observing Samp1.

v = L \ KxSamp1x;
Kxx = sigmaSq_f .* CF_SquaredExponential(x, x, l);
covFuncPostSamp1Atx = Kxx - v'*v; % Posterior covariance function @ x after observing Samp1.
varFuncPostSamp1Atx = diag(covFuncPostSamp1Atx);

plot(x, mFuncPostSamp1Atx , 'b--')
patch([x ; flipud(x)], [mFuncPostSamp1Atx + (2*sqrt(varFuncPostSamp1Atx)) ; ...
    flipud( mFuncPostSamp1Atx - (2*sqrt(varFuncPostSamp1Atx)))] ,...
    [160,160,160]./255 , 'FaceAlpha' , 0.3 , 'LineStyle' , 'none')

%% Non-Implausible Set

impThresh = 2; % Implausibility threshold. Chosen to be 2 so it matches visually wth the +-2 SD error bounds.
nonImpLogic = ((yTest - mFuncPostSamp1Atx) .^2 ./ varFuncPostSamp1Atx) <= impThresh^2;
nonImpSet = x(nonImpLogic);
plot([x(1), x(end)] , [yTest, yTest] , 'r--')
plot(nonImpSet, yTest , 'r.')

%% Sample

xSamp2 = round(lhsdesign(nSamp,1).*length(nonImpSet));
xSamp2 = nonImpSet(xSamp2);
xSamp2 = sort([xSamp1 ; xSamp2]);
ySamp2 = f(xSamp2);

figure;
plot(x,y , 'k');
xlabel('x')
ylabel('y')
title('2nd Iteration')
hold on

plot(xTest, yTest, 'kx')
plot(xSamp2, ySamp2 , 'bx')

%% GP

KxSamp2xSamp2 = sigmaSq_f .* CF_SquaredExponential(xSamp2, xSamp2, l) + eye(length(xSamp2)) .* sigmaSq_n;
L = chol(KxSamp2xSamp2 , 'lower');
alpha = L'\(L\ySamp2);

KxSamp2x = sigmaSq_f .* CF_SquaredExponential(xSamp2, x, l);
mFuncPostSamp2Atx = KxSamp2x' * alpha; % Posterior mean function @ x after observing Samp2.

v = L \ KxSamp2x;
covFuncPostSamp2Atx = Kxx - v'*v; % Posterior covariance function @ x after observing Samp2.
varFuncPostSamp2Atx = diag(covFuncPostSamp2Atx);

plot(x, mFuncPostSamp2Atx , 'b--')
patch([x ; flipud(x)], [mFuncPostSamp2Atx + (2*sqrt(varFuncPostSamp2Atx)) ; ...
    flipud( mFuncPostSamp2Atx - (2*sqrt(varFuncPostSamp2Atx)))] ,...
    [160,160,160]./255 , 'FaceAlpha' , 0.3 , 'LineStyle' , 'none')

%% Non-Implausible Set

nonImpLogic = ((yTest - mFuncPostSamp2Atx) .^2 ./ varFuncPostSamp2Atx) <= impThresh^2;
nonImpSet = x(nonImpLogic);
plot([x(1), x(end)] , [yTest, yTest] , 'r--')
plot(nonImpSet, yTest , 'r.')

%% Sample

xSamp3 = round(lhsdesign(nSamp,1).*length(nonImpSet));
xSamp3 = nonImpSet(xSamp3);
xSamp3 = sort([xSamp2 ; xSamp3]);
ySamp3 = f(xSamp3);

figure;
plot(x,y , 'k');
xlabel('x')
ylabel('y')
title('3rd Iteration')
hold on

plot(xTest, yTest, 'kx')
plot(xSamp3, ySamp3 , 'bx')

%% GP

KxSamp3xSamp3 = sigmaSq_f .* CF_SquaredExponential(xSamp3, xSamp3, l) + eye(length(xSamp3)) .* sigmaSq_n;
L = chol(KxSamp3xSamp3 , 'lower');
alpha = L'\(L\ySamp3);

KxSamp3x = sigmaSq_f .* CF_SquaredExponential(xSamp3, x, l);
mFuncPostSamp3Atx = KxSamp3x' * alpha; % Posterior mean function @ x after observing Samp3.

v = L \ KxSamp3x;
covFuncPostSamp3Atx = Kxx - v'*v; % Posterior covariance function @ x after observing Samp3.
varFuncPostSamp3Atx = diag(covFuncPostSamp3Atx);

plot(x, mFuncPostSamp3Atx , 'b--')
patch([x ; flipud(x)], [mFuncPostSamp3Atx + (2*sqrt(varFuncPostSamp3Atx)) ; ...
    flipud( mFuncPostSamp3Atx - (2*sqrt(varFuncPostSamp3Atx)))] ,...
    [160,160,160]./255 , 'FaceAlpha' , 0.3 , 'LineStyle' , 'none')

%% Non-Implausible Set

nonImpLogic = ((yTest - mFuncPostSamp3Atx) .^2 ./ varFuncPostSamp3Atx) <= impThresh^2;
nonImpSet = x(nonImpLogic);
plot([x(1), x(end)] , [yTest, yTest] , 'r--')
plot(nonImpSet, yTest , 'r.')