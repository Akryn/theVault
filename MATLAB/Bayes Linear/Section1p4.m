%% Michael Goldstein & David Wooff - Bayes Linear Statistics (2007)
% Chapter 1.4

%% 1.4.2 Prior Inputs

% X_1 and X_2 are the sales of the products 1 and 2 at the first time point.
% Y_1 and Y_2 are the sales of the products at the later time point.

E_X1 = 100;
E_X2 = 100;
E_Y1 = 100;
E_Y2 = 100;
% We have the same expectation for sales for each product at each time point.

Var_X1 = 25;
Var_X2 = 25;
Var_Y1 = 100;
Var_Y2 = 100;
% we are less certain about the sales for the later time point.

Corr = [...
    1, -0.6, 0.6, -0.2;...
    -0.6, 1, -0.2, 0.6;...
    0.6, -0.2, 1, -0.6;...
    -0.2, 0.6, -0.6, 1]; % Corr([X1; X2; Y1; Y2], [X1; X2; Y1; Y2])
% Corr matrix expresses the belief that the sales of each product are quite strongly positvely
% correlated over the two time periods, but that the products are considered to compete and so sales
% of the two products are negatively correlated.

% We intent to use the sales at the first time point to improve our forecasts for sales at the later
% time point.

% Much of our approach deals with simultaneous analysis of collections of quantities, so for
% convenience, we group together the two sales from the first time point into the collection D =
% [X1; X2], and the sales for the later time point into the collection B = [Y1; Y2].
% D is the collection of "data" quantities (i.e. quantities we intend to observe, and so for which
% data will become available) and to retain B for a collection of "belief" quantities (i.e.
% quantities that we wish to predict, and so for which we have prior beliefs followed by adjusted
% beliefs).

% I will use D but often use Y instead of B.
E_D = [E_X1; E_X2];
E_B = [E_Y1; E_Y2];
Var_D = diag(sqrt([Var_X1; Var_X2])) * Corr(1:2, 1:2) * diag(sqrt([Var_X1; Var_X2]));
Var_B = diag(sqrt([Var_Y1; Var_Y2])) * Corr(3:4, 3:4) * diag(sqrt([Var_Y1; Var_Y2]));
Cov_Y1D = sqrt(Var_Y1) * Corr(3, 1:2) * diag(sqrt([Var_X1; Var_X2]));
Cov_Y2D = sqrt(Var_Y2) * Corr(4, 1:2) * diag(sqrt([Var_X1; Var_X2]));
Cov_BD = diag(sqrt([Var_Y1; Var_Y2])) * Corr(3:4, 1:2) * diag(sqrt([Var_X1; Var_X2]));

VarCov = diag(sqrt([Var_X1; Var_X2; Var_Y1; Var_Y2])) * Corr * diag(sqrt([Var_X1; Var_X2; Var_Y1; Var_Y2]));

%% 1.4.3 Adjusted Expectation (Random Variable)
% Look among the collection of linear estimates, i.e. those of the form c_0 + c_1X1 + c_2X2, to
% minimise the prior expected squared error loss in estimating Y.
% For example, for Yi, we aim to minise E([Yi - c_0 - c_1X1 - c_2X2]).
% This is given by the Bayes linear rule for Y given X:
% E_D(Y1) = E(Y1) + Cov(Y1,D) * Var(D)^-1 * (D - E(D))    See below for proof
%         = 100 + ([0.6, -0.2] * (sqrt(100) * sqrt(25))) * pinv([25, -15; -15, 25]) * (D - 100)
%         = 100 + [30, -10] * [0.0625, 0.0375; 0.0375, 0.0625] * ([X1; X2] - [100; 100])
%         = 100 + [1.5, 0.5] * ([X1; X2] - [100; 100])
%         = 100 + 1.5*X1 + 0.5*X2 - 1.5*100 - 0.5*100
%         = 1.5X1 + 0.5*X2 - 100
% We call E_D(Y1) the adjusted expectation of Y1 given D.
% It is itself a random quantity since it is a function of the random variable D and so has an
% expectation, variance and so forth.

% Proof of Bayes Linear Update Rule:
% Want h = [h*; h0] which minimises C = E[(Yi - h*'D - h0)^2]
% Notice first that for any h*, the value of h0 which minises C is h0 = E(Yi) - (h*'D) = E(Yi) - E(h*'D)
% This is because E((X - M)^2) is minised when M = E(X).
% See John K. Kruske - Doing Bayesian Data Analysis (2nd Edition) 4.3.1.1 Mean as minimized variance
% page 86 for an explanation on this.
% With this choice of h0, C = E[(Yi - h*'D - E(Yi - h*'D))^2]
%                           = Var(Yi - h*'D)
%                           = Var(Yi) + Var(h*'D) - 2Cov(Yi, h*'D)
%                           = Var(Yi) + h*'Var(D)h* - 2Cov(Yi, D)h*
% C is then minimised when dC/dh* = 0
% dC/dh* = 2Var(D)h* - 2Cov(D, Yi)
% C is minimised when h* = Var(D)^-1 * Cov(D, Yi)
% Therefore, E_D(Y) = h0 + h*'D
%                   = E(Yi) - E(h*'D) + Cov(Yi, D)Var(D)^-1 * D
%                   = E(Yi) - E(Cov(Yi, D)Var(D)^-1 * D) + Cov(Yi, D)Var(D)^-1 * D
%                   = E(Yi) - Cov(Yi, D)Var(D)^-1 * E(D) + Cov(Yi, D)Var(D)^-1 * D
%                   = E(Yi) + Cov(Yi, D)Var(D)^-1 * (D - E(D))

%% 1.4.4 Adjusted Versions (Random Variable)
% The associated residual.
% A_D(Yi) = Yi - E_D(Yi)
% This is also a random quantity since it relies on Yi and E_D(Yi) which are random.
% A priori, we expect this to be zero, E(A_D(Yi)) = 0

%% 1.4.5 Adjusted Variances (NOT Random in 2nd order analysis)
% How useful are the adjusted expectations as predictors? We can evaluate the adjusted variance for
% each quantiy which is DEFINED as the variance of the adjusted version.
% Var_D(Yi) := Var(A_D(Yi)) = E([Yi - E_D(Yi)]^2)
% which, due to the definition of the adjusted expectation, is the minimum of the prior expected
% squared error loss.
% This is a measure of uncertainty, or, informally, the "unexplained" variance, having taken into
% account the information in D.
% This is given by:
% Var_D(Yi) = Var(Yi) - Cov(Yi,D) * Var(D)^-1 * Cov(D,Yi)    See below for derivation
Var_D_Y1 = Var_Y1 - Cov_Y1D * pinv(Var_D) * Cov_Y1D';
% 60
Var_D_Y2 = Var_Y2 - Cov_Y2D * pinv(Var_D) * Cov_Y2D';
% 60

% The portion of variance resolved is:
% Var(E_D(Yi)) = Var(Yi) - Var_D(Yi)
Var_E_D_Y1 = Var_Y1 - Var_D_Y1;
% 40
Var_E_D_Y2 = Var_Y2 - Var_D_Y2;
% 40
% So, by observing the sales at the first time point, we expect to reduce our uncertainty on the
% sales at the later time point by 40%.

% We typically summarise the informativeness of the data for any quantity Yi by a scale-free measure
% called the resolution of Yi induced by D. This lies between 0 and 1.
% R_D(Yi) = 1 - (Var_D(Yi) / Var(Yi)) = Var(E_D(Yi)) / Var(Yi)
R_D_Y1 = Var_E_D_Y1 ./ Var_Y1;
% 0.4
R_D_Y2 = Var_E_D_Y2 ./ Var_Y2;
% 0.4

% {Small, Large} resolutions imply that the information has {little, much} linear predictive value,
% given the prior specification.

% In terms of the collection of Ys, B, we have decomposed as follows:
% Var(B) = RVar_D(B) +  Var_D(B)
% where (DEFINITION) RVar_D(B) := Var(E_D(B)) is the notation for the resolved matrix for the
% adjustment of the collection B by the collection D, and equals the prior variance matrix for the
% adjusted expectation vector.

Var_D_B = Var_B - Cov_BD * pinv(Var_D) * Cov_BD';
% [60, -60; -60, 60]

RVar_D_B = Var_B - Var_D_B;
% [40, 0; 0, 40]
%% 1.4.6 Checking Data Inputs
% Now we observe some values of D, d = [x1; x2].
x1 = 109;
x2 = 90.5;
d = [x1; x2];

% First check that these observations are consistent with beliefs specified about them beforehand.
% Examine the standarised change from the prior expectation to the observed value.
S_x1 = (x1 - E_X1) ./ sqrt(Var_X1);
% 1.8
S_x2 = (x2 - E_X2) ./ sqrt(Var_X2);
% -1.9

% Each squared standardised change has prior expectation 1.
% E(S_Xi^2) = E([Xi - E(Xi)]^2 ./ Var(Xi)])
%           = E([Xi - E(Xi)]^2) ./ Var(Xi)   as Var(Xi) is a constant.
%           = Var(Xi) ./ Var(Xi)
%           = 1
% We might begin to suspect an inconsistency if we saw a standardised change of more than 2 standard
% deviations, and quite concerned to see standardised changes of more than about 3 standard
% deviations.

%% 1.4.7 Observed Adjusted Expectations
% Replace D by d in the 1.4.3.

% E_d(Yi) = E(Yi) + Cov(Yi,D) * Var(D)^-1 * (d - E(D))
E_d_Y1 = E_Y1 + Cov_Y1D * pinv(Var_D) * (d - E_D);
% 108.75
E_d_Y2 = E_Y2 + Cov_Y2D * pinv(Var_D) * (d - E_D);
% 90.25

% To calculate for the collection B,
% E_d(B) = E(B) + Cov(B,D) * Var(D)^-1 * (d - E(D))
E_d_B = E_B + Cov_BD * pinv(Var_D) * (d - E_D);

%% 1.4.8 Diagnostics for Adjusted Beliefs (Standardised Adjustments)
% At this stage, we check how different the observed adjusted expectation is from the prior
% expectation. A simple diagnostic is given by the change from prior to adjusted expectation,
% standardised with respect to the variance of the adjusted expectation.
% S(E_d(Yi)) = (E_d(Yi) - E(E_D(Yi))) ./ sqrt(Var(E_D(Yi)))
%           = (E_d(Yi) - E(Yi)) ./ sqrt(Var(E_D(Yi)))    as E(E_D(Yi)) = E(Yi).    See below for proof
S_E_d_Y1 = (E_d_Y1 - E_Y1) ./ sqrt(Var_E_D_Y1);
% 1.3835
S_E_d_Y2 = (E_d_Y2 - E_Y2) ./ sqrt(Var_E_D_Y2);
% -1.5416

% These values have prior expectation 1 and so are roughly in line with what we expected beforehand.

% Proof E(E_D(Y)) = E(Y):
% E(E_D(Y)) = E[E(Y) + Cov(Y, D)Var(D)^-1 * (D - E(D))]
%           = E[E(Y)] + E[Cov(Y, D)Var(D)^-1 * (D - E(D))]
%           = E(Y) + Cov(Y, D)Var(D)^-1 * E[(D - E(D))]
%           = E(Y) + Cov(Y, D)Var(D)^-1 * [E(D) - E(E(D))]
%           = E(Y) + Cov(Y, D)Var(D)^-1 * [E(D) - E(D)]
%           = E(Y)

%% 1.4.9 Further Diagnostics for the Adjusted Versions
y1 = 112;
y2 = 95.5;
b = [y1; y2];
% It's important to compare our predictions to what actually happened.
% 1). We can compare a quantity's observation with it's prior expectation, irrespective of the linear
% fitting on D.
% S(yi) = (yi - E(Yi)) ./ sqrt(Var(Yi))
S_y1 = (y1 - E_Y1) ./ sqrt(Var_Y1);
% 1.2
S_y2 = (y2 - E_Y2) ./ sqrt(Var_Y2);
% -0.45
% The sales were consistent with our prior considerations.

% 2). We can examine the change from adjusted expectation to the observation, relative to the
% adjusted variance, as this was the variation remaining in Y after fitting X, but before observing
% Y.
% By obsering y, we observe the residual component, i.e. the adjusted version random variables A_D(Y) = Y - E_D(Y).
A_d_y1 = y1 - E_d_Y1;
% 3.25
A_d_y2 = y2 - E_d_Y2;
% 5.25
% Given that they had prior expectation 0, we wish to see how far the adjusted versions have changed
% from 0.
% S_d(yi) := S(A_d(yi))    i.e. S_d(y) is DEFINED to be S(A_d(yi)).
%          = {A_d(yi) - E(A_D(Yi))} ./ sqrt(Var(A_D(Yi)))
%          = {(yi - E_d(Yi)) - E(Yi - E_D(Yi))} ./ sqrt(Y - E_D(Yi))
%          = {(yi - E_d(Yi)) - E(Yi) + E(E_D(Yi))} ./ sqrt(Var_D(Yi))    as Var_D(Yi) is defined as Var(Yi - E_D(Yi))
%          = {(yi - E_d(Yi)) - E(Yi) + E(Yi)} ./ sqrt(Var_D(Yi))    as E(E_D(Yi)) = E(Yi)
%          = (yi - E_d(Yi)) ./ sqrt(Var_D(Yi))    This shows adjusted versions have prior expectation 0.
S_d_y1 = (y1 - E_d_Y1) ./ sqrt(Var_D_Y1);
% 0.4196
S_d_y2 = (y2 - E_d_Y2) ./ sqrt(Var_D_Y2);
% 0.6678
% These should again be about 1, so our checks were roughly within the tolerances suggested by our
% prior variance specifications. If anything, the adjusted expectations are, in terms of standard
% deviations, rather closer closes to the observed values than expected.

% We typically write S_d(yi) and S(A_d(yi)).
% This is useful notation to show variables outside the brackets don't get turned in to random variables in
% the subtracted expectation.
% e.g. S_d(yi)    = yi      - E_d(Yi)    ./ sqrt(Var_D(Yi))       Observed d in - E_d(Yi)
%      S(A_d(yi)) = A_d(Yi) - E(A_D(Yi)) ./ sqrt(Var(A_D(Yi)))    Random variable D in - E(A_D(Yi))

%% Summary of Basic Adjustment (Part 1)
% Here, the book shows a table. I think I will understand this better with a visualisation of
% everything which has gone on so far and so will plot the results of sampling our example.

% Randomly sample X1, X2, Y1 and Y2 using
n = 100000;
X1 = zeros(n,1);
X2 = zeros(n,1);
Y1 = zeros(n,1);
Y2 = zeros(n,1);

for a = 1: n
    r = chol(VarCov,'lower') * randn(4,1) + [E_D; E_B];
    
    X1(a) = r(1);
    X2(a) = r(2);
    Y1(a) = r(3);
    Y2(a) = r(4);
end

mean(X1); % ~ 100
mean(X2); % ~ 100
mean(Y1); % ~ 100
mean(Y2); % ~ 100

var(X1); % ~ 25
var(X2); % ~ 25
var(Y1); % ~ 100
var(Y2); % ~ 100

E_D_Y1 = 1.5 .* X1 + 0.5 .* X2 - 100;
E_D_Y2 = 0.5 .* X1 + 1.5 .* X2 - 100;

mean(E_D_Y1); % ~ 100
mean(E_D_Y2); % ~ 100

var(E_D_Y1); % ~ 40
var(E_D_Y2); % ~ 40

A_D_Y1 = Y1 - E_D_Y1;
A_D_Y2 = Y2 - E_D_Y2;

mean(A_D_Y1); % ~ 0
mean(A_D_Y2); % ~ 0

var(A_D_Y1); % ~ 60
var(A_D_Y2); % ~ 60

clear a ans r

%% Summary of Basic Adjustment (Part 2)

stepSize = n/100;
figure;
subplot(2,2,1)
plot(1:100, Y1(1:stepSize:end), 'gx')
hold on
plot([1,100], [E_Y1,E_Y1], 'k')
for a = 1 : 100
    plot([a,a], [Y1(1 + (a-1)*stepSize) , E_Y1], 'k')
end
title('Comparing Y1 & Prior Expectation E(Y1)')

subplot(2,2,3)
plot(1:100, Y1(1:stepSize:end), 'gx')
hold on
plot([1,100], [E_Y1,E_Y1], 'k')
plot(1:100, E_D_Y1(1:stepSize:end), 'b.')
for a = 1 : 100
    plot([a,a], [ E_D_Y1(1 + (a-1)*stepSize), Y1(1 + (a-1)*stepSize)], 'r')
end
title('Comparing Y1 & Adjusted Expectation E_D(Y1)')


subplot(2,2,2)
plot(1:100, Y2(1:stepSize:end), 'gx')
hold on
plot([1,100], [E_Y2,E_Y2], 'k')
for a = 1 : 100
    plot([a,a], [Y2(1 + (a-1)*stepSize) , E_Y2], 'k')
end
title('Comparing Y2 & Prior Expectation E(Y2)')

subplot(2,2,4)
plot(1:100, Y2(1:stepSize:end), 'gx')
hold on
plot([1,100], [E_Y2,E_Y2], 'k')
plot(1:100, E_D_Y2(1:stepSize:end), 'b.')
for a = 1 : 100
    plot([a,a], [ E_D_Y2(1 + (a-1)*stepSize), Y2(1 + (a-1)*stepSize)], 'r')
end
title('Comparing Y2 & Adjusted Expectation E_D(Y2)')

% Notice that the red verticle lines are typically shorter than the black ones (not always - some
% red lines cross through the black horizontal line of E(Yi), the prior expectation).

clear stepSize

%% Summary of Basic Adjustment (Part 3)

edges = 50:150;
Aedges = -50:50;

figure;
subplot(3,2,1)
histogram(Y1, edges, 'FaceColor' , 'g')
hlim = ylim;
title(['Y1  |  mean(Y1) = ' , num2str(mean(Y1)) , ' (100)  |  var(Y1) = ' , num2str(var(Y1)) , ' (100)'])

subplot(3,2,3)
histogram(E_D_Y1, edges, 'FaceColor' , 'b')
hlim(3,:) = ylim;
title(['E_D(Y1) = 1.5X1 + 0.5X2 - 100  | mean(E_D(Y1)) = ' , num2str(mean(E_D_Y1)) , ' (100)  |  var(E_D(Y1)) = ' , num2str(var(E_D_Y1)) , ' (40)'])

subplot(3,2,5)
histogram(A_D_Y1, Aedges, 'FaceColor' , 'r')
hlim(5,:) = ylim;
title(['A_D(Y1) = Y - E_D(Y1)  |  mean(A_D(Y1)) = ' , num2str(mean(A_D_Y1)) , ' (0)  |  var(A_D(Y1)) = ' , num2str(var(A_D_Y1)) , ' (60)'])


subplot(3,2,2)
histogram(Y2, edges, 'FaceColor' , 'g')
hlim(2,:) = ylim;
title(['Y2  |  E(Y2) = ' , num2str(mean(Y2)) , ' (100)  |  Var(Y2) = ' , num2str(var(Y2)) , ' (100)'])

subplot(3,2,4)
histogram(E_D_Y2, edges, 'FaceColor' , 'b')
hlim(4,:) = ylim;
title(['E_D(Y2) = 0.5X1 + 1.5X2 - 100  |  mean(E_D(Y2)) = ' , num2str(mean(E_D_Y2)) , ' (100)  |  var(E_D(Y2)) = ' , num2str(var(E_D_Y2)) , ' (40)'])

subplot(3,2,6)
histogram(A_D_Y2, Aedges, 'FaceColor' , 'r')
hlim(6,:) = ylim;
title(['A_D(Y2) = Y - E_D(Y2)  |  mean(A_D(Y2)) = ' , num2str(mean(A_D_Y2)) , ' (0)  |  var(A_D(Y2)) = ' , num2str(var(A_D_Y2)) , ' (60)'])

maxlim = max(hlim);

for a = 1:6
    subplot(3,2,a)
    ylim(maxlim)
end

clear n X1 X2 Y1 Y2 E_D_Y1 E_D_Y2 A_D_Y1 A_D_Y2 a edges Aedges hlim maxlim

%% 1.4.11 Diagnostics for Collections
% Up to now, we have been checking individual data inputs by calculating the standardised changes.
% To check a collection of data inputs, we need to make a basic consistency check, and if that is
% successful, we proceed to calculate a global discrepancy.

% For any quantity X, if we specify Var(X) = 0, then we should observe x = E(X), otherwise either
% our specification is wrong or there was error in collecting the data.
% The basic consistency check is as follows:
% If Var(B) is non-singular (it is invertible), then the value of b - E(B) is unconstrained, and the
% basic consistency check is passed. Otherwise Var(B) has one or more eigenvalues equal to 0.
% In this case, suppose q is an eigenvector corresponding to a 0 eigenvalue. Such eigenvectors
% identify linear combinations of the Bs having zero variance, as for each such eigenvector q, it is
% the case that Var(q'B) = 0.

% For our measure of the difference between the data d and the prior expectations E(D), we DEFINE
% the discrepancy, Dis(d), as the Mahalanobis distance between d and E(D).
% Dis(d) := (d - E(D))' * pinv(Var(D)) * (d - E(D))
Dis_d = (d - E_D)' * pinv(Var_D) * (d - E_D);
% 4.2906

% The discrepancy has prior expectation equal to the rank of the prior variance matrix Var(D), which
% in our example has rank 2.
rk_Var_D = rank(Var_D);
% 2

% The discrepancy between the observed values and the prior specification can be summarised as the
% discrepancy ratio, DEFINED as:
% Dr(d) := Dis(d) / rk(Var(D))
% and has prior expectation 1.
Dr_d = Dis_d / rk_Var_D;
% 2.1453

% For single observations, rather than collections, the discrepancies are just the squared
% standardised changes.

% Similar to the above, we obtain a global diagnostic for the difference between the observed
% adjusted expectation and the prior (adjusted) expectation by calculating the Mahalanobis distance between the
% two to give the adjustment discrepancy:
% Dis_d(B) := (E_d(B) - E(E_D(B)))' * pinv(RVar_D(B)) * (E_d(B) - E(E_D(B)))
%           = (E_d(B) - E(B))' * pinv(RVar_D(B)) * (E_d(B) - E(B))
Dis_d_B = (E_d_B - E_B)' * pinv(RVar_D_B) * (E_d_B - E_B);
% 4.2906

% Again, similarly, we obtain a global diagnostic for the difference from the actual observation to
% the observed adjusted expectation with respect to the remaining variance before observing it.
% Another way of thinking about this is that we finally observe A_D(B) and wish to see whether these
% observations are cosistent with their prior variance-covariance specifications, Var(A_D(B)).
% For our basic cosistency check, we discover Var(A_D(B)) = Var_D(B) is singular.
rk_Var_D_B = rank(Var_D_B);
% 1
% There is 1 eigenvalue equal to 0, with the corresponding eigenvector proportional to [1; 1].
% Consequently we have specified a variance of zero for [1, 1][A_D(Y1); A_D(Y2)].
% and it is thus necessary to verify in this example that the observed adjusted versions sum to
% their expected value (because the eigenvector was proportional to [1, 1], they must sum).
% However, A_d(y1) + A_d(y2) = 3.25 + 5.25 = 8.5 ~= 0, so we have discovered a serious flaw in our
% specification. In practice, there is no point in proceeding further with the analysis.

% Had the basic consistency check not failed, we would have calculated the adjusted version
% discrepancy as
% Dis(A_d(b)) = {A_d(b) - E(A_D(B))}' * pinv(Var(A_D(B))) * {A_d(b) - E(A_D(B))}
%             = {(b - E_d(B)) - (E[B - E_D(B)])}' * pinv(Var_D(B - E_D(B))) * {(b - E_d(B)) - (E[B - E_D(B)])}
%             = {(b - E_d(B)) - E(B) + E(E_D(B))}' * pinv(Var_D(B)) * {(b - E_d(B)) - E(B) + E(E_D(B))}    as Var_D(B) is defined as Var_D(B - E_D(B))
%             = {(b - E_d(B)) - E(B) + E(B)}' * pinv(Var_D(B)) * {(b - E_d(B)) - E(B) + E(B)}    as E(E_D(B)) = E(B)
%             = (b - E_d(B))' * pinv(Var_D(B)) * (b - E_d(B))
Dis_A_d_b = (b - E_d_B)' * pinv(Var_D_B) * (b - E_d_B);
% 0.0167

% Actual observations which should not have been possible given the prior specifications suggests we
% should focus on analysing collections of belief, rather than on piecemeal analysisfor single
% quantities.

%% 1.4.12 Exploring Collections of Beliefs via Canonical Structure
% Whether our interest is in making assessments for simple quantities such as Y1, or for interesting
% linear combinations such as Y1 + Y2, or for global collections such as B = [Y1; Y2], there is a
% natrual reorganisation which we may use to answer these questions directly. The reorganisation
% arises by generating and exploiting an underlying canonical structure which completely summarises
% the global dynamics of belief adjustment for an anlysis. For the 2 dimension problem, this amounts
% to finding the linear combinations of Y1 and Y2 about which D is respectively most and least
% informative, in the sense of maximising and minising the variance resolution.

% The jth canonical resolution for the adjustment of B by D is the jth largest eigenvalue of the
% matrix
% The jth canonical direction for the adjustment of B by D is the linear combination h'B where h is
% the eigenvector corresponding to the jth largest eigenvalue (which is the jth cacnonical resolution)
% of the matrix pinv(Var(B)) * RVar_D(B), scaled to prior expectation zero 0 and variance 1.
[eigvec, eigval] = eig(pinv(Var_B) * RVar_D_B);
% h1 is calculated as 1/2[sqrt(2); sqrt(2)] by MATLAB but I've taken it to equivalently be [1; 1].
% Z1 = {(h1' * B) - E((h1' * B))} / Var((h1' * B))
%    = (Y1 + Y2) - (100 + 100) / Var(Y1 + Y2)
%    = Y1 + Y2 - 200 / sqrt([Var(Y1) + Var(Y2) + 2Cov(Y1,Y2)])
%    = Y1 + Y2 - 200 / sqrt((100 + 100 + 2(-60)))
%    = Y1 + Y2 - 200 / sqrt(80)
%    = 0.1118Y1 + 0.1118Y2 - 22.3607
% h2 is calculated as 1/2[-sqrt(2); sqrt(2)] by MATLAB but I've taken it to equivalently be [1; -1].
% Z2 = {(h1' * B) - E((h1' * B))} / Var((h1' * B))
%    = (Y1 - Y2) - (100 - 100) / Var(Y1 + Y2)
%    = (Y1 - Y2) / sqrt([Var(Y1) + Var(Y2) - 2Cov(Y1,Y2)])
%    = (Y1 - Y2) / sqrt((100 + 100 - 2(-60)))
%    = (Y1 - Y2) / sqrt(320)
%    = (Y1 - Y2) / 8sqrt(5)
%    = 0.0559Y1 - 0.0559Y2
disp('Derivation of canonical structure formula.')
% Z1 and Z2 are called the first and second canonical directions respectively.
% Canonical directions are always uncorrelated.
disp('Why are canonical directions always uncorrelated?')

% The original sales quantities Yi can then be expressed in terms of the canonical quantities:
% Y1 = 4.472(Z1 + 2Z2) + 100
% Y2 = 4.472(Z1 - 2Z2) + 100

% The resolutions are:
% R_D(Zi) = lambdai
R_D_Z1 = eigval(1,1);
% 1
R_D_Z2 = eigval(2,2);
% 0.25
% The latter implies the minimum variance resolution for ANY linear combination of the two unknown
% sales quantities is 0.25, i.e. by observing D we explect to "explain" at least 25% of the variance
% for ALL linear combinations.
% The resolution of Z1 turns out to be exactly 1 meaning that, according to our prior
% specifications, there will be no uncertainty remaining in Z1 once we have observed previous sales
% X1 and X2.
% As Z1 is proportional (except for a constant) to total sales: Y1 + Y2 = 8.944Z1 + 200,
% we shall have no uncertainty about Y1 + Y2 after we have observed X1 and X2.
% E_d(Y1) + E_d(Y2) = 108.75 + 90.25 = 199 FOR CERTAIN.
% We most likely did not intend our prior specifications to contain the algebraic implication that
% we will "know" total future sales in advance. We end up observing y1 + y2 = 112 + 95.5 = 207.5
% which contradicts the prior specification and reulted in the failure of the consistency check in
% the last section.

% What led to this? Calculate the adjusted expectations of Z1 and Z2.
% We introduce notation for the main sums and differences.
% Xp = X1 + X2
% Xm = X1 - X2
% Yp = Y1 + Y2
% Ym = Y1 - Y2

% E_D(Z1) = E(Z1) + Cov(Z1,D) * pinv(Var(D)) * (D - E(D))
%         = E((Yp - 200) / sqrt(80)) + Cov(Yp - 200,D) * pinv(Var(D)) * (D - E(D))
%         = (100 + 100 -200)/sqrt(80) + Cov((Yp - 200) / sqrt(80), D) * pinv(Var(D)) * (D - E(D))
%         = Cov((Yp - 200) / sqrt(80), D) * pinv(Var(D)) * (D - E(D))
%         = {Cov(Y1/sqrt(80),D) + Cov(Y2/sqrt(80),D)} * pinv(Var(D)) * (D - E(D))
%         = {Cov(Y1/sqrt(80),D) + Cov(Y2/sqrt(80),D)} * pinv(Var(D)) * (D - E(D))
%         = 1/sqrt(80){Cov(Y1,D) + Cov(Y2,D)} * pinv(Var(D)) * (D - E(D))
%         = 1/sqrt(80){[30,-10] + [-10,30]} * pinv(Var(D)) * (D - [100; 100])
%         = 1/sqrt(80)[20, 20] * [0.0625, 0.0375; 0.0375, 0.0625] * ([X1; X2] - [100; 100])
%         = 1/sqrt(80)[2, 2] * ([X1; X2] - [100; 100])
%         = 1/sqrt(80)[2, 2] * ([X1; X2] - [100; 100])
%         = 1/sqrt(20)[1, 1] * ([X1; X2] - [100; 100])
%         = 1/sqrt(20)[X1 + X2] - 200/sqrt(20)
%         = Xp/sqrt(20) - 10sqrt(20)
%         = Xp/sqrt(20) - 20sqrt(5)
%         = 0.2236Xp - 44.7214

% E_D(Z2) = E(Z2) + Cov(Z2,D) * pinv(Var(D)) * (D - E(D))
%         = E(Ym/8sqrt(5)) + Cov(Ym/8sqrt(5),D) * [0.0625, 0.0375; 0.0375, 0.0625] * ([X1; X2] - [100; 100])
%         = 1/8sqrt(5){ Cov(Y1,D) - Cov(Y2,D)} * [0.0625, 0.0375; 0.0375, 0.0625] * ([X1; X2] - [100; 100])
%         = 1/sqrt(8)[40, -40] * [0.0625, 0.0375; 0.0375, 0.0625] * ([X1; X2] - [100; 100])
%         = 1/sqrt(8)[1, - 1] * ([X1; X2] - [100; 100])
%         = 1/sqrt(8)(X1 - X2 -100 +100)
%         = 1/sqrt(8)Xm

% Resolution R_D(Z1) = 1 corresponds to having adjusted variance of zero for E_D(Z1), so the
% correlation between Z1 (proportional to Yp) and E_D(Z1) (proportional to Xp) must be equal to 1.
% Thus, Xp and Yp have a prior correlation of one, and this explains why Yp becomes "known" as soon
% as we observe xp.

%% 1.4.13 Modifying the Original Specifications
% We wish to not change our prior means and variances for the four sales quantities, but just to
% weaken one or two of the correlations.

% In terms of the four sums and differences, the original prior correlation matrix was:
%    1    0   0.5   0
%    0    1    0    1
%   0.5   0    1    0
%    0    1    0    1

% Suppose we decide that it is appropriate to weaken the correlation between Xp and Yp to 0.8.
% The prior correlation matrix over sales then becomes:
%     1    -0.6   0.56  -0.24
%   -0.6    1    -0.24   0.56
%   0.56  -0.24    1     -0.6 
%  -0.24   0.56   -0.6     1
% so the actual effect is to decrease generally all the correlations between the sales quantities.