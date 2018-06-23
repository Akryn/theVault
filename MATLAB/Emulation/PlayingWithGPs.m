x = [0:0.1:10]';
y = sin(x);

clf
subplot(1,2,1)
TruePlot = plot(x , y , 'k');
hold on
PriorMeanPlot = plot(x , zeros(length(x) , 1) , 'k--');

patch([0 , 0 , 10 , 10] , [-2 , 2 , 2 , -2] , [160,160,160]./255 , 'FaceAlpha' , 0.3 , 'LineStyle' , 'none')
xlim([0,10])
ylim([-3 , 3])

nugget = 0.000001;
% Prior GP: 0 mean, Sqaured exponential Covariance with correlation length 1 and sigma_n = 0.000001
KXX = CF_SquaredExponential(x,x, 1);
L = chol(KXX + (nugget .* eye(length(x)) ) , 'lower');

% Sample Prior
norm1 = randn(length(x) , 1);
samp1 = 0 + L * norm1;

norm2 = randn(length(x) , 1);
samp2 = 0 + L * norm2;

norm3 = randn(length(x) , 1);
samp3 = 0 + L * norm3;



plot(x , samp1 , 'r--')
plot(x , samp2 , 'g--')
plot(x , samp3 , 'b--')

legend([TruePlot , PriorMeanPlot] , 'True Function' , 'Prior Mean Function')
title('Prior')
xlabel('x')
ylabel('y')

% Update

x_train = [2 ; 5 ; 7 ; 9];
y_train = sin(x_train);

KXtrainXtrain = CF_SquaredExponential(x_train,x_train, 1);
L = chol(KXtrainXtrain + (nugget .* eye(length(x_train)) ) , 'lower');

alpha = L' \ (L\y_train);

KXstarXtrain = CF_SquaredExponential(x , x_train, 1);
f_star = KXstarXtrain * alpha;

v = L \ KXstarXtrain'; % Need KXtrainXstar which is the same as KXstarXtrain'
KXstarXstar = CF_SquaredExponential(x , x , 1);
Vf_star = KXstarXstar - (v'*v);


% Sample Posterior
L = chol(Vf_star + (nugget .* eye(length(x)) )  , 'lower');

updated_samp1 = f_star + L * norm1;

updated_samp2 = f_star + L * norm2;

updated_samp3 = f_star + L * norm3;

subplot(1,2,2)

TruePlot = plot(x , y , 'k');
hold on
TrainingPlot = plot(x_train , y_train , 'kx');

PosteriorMeanPlot = plot(x , f_star , 'k--');

patch([x ; flipud(x)], [f_star + (2*sqrt(diag(Vf_star))) ; flipud( f_star - (2*sqrt(diag(Vf_star))))] ,...
    [160,160,160]./255 , 'FaceAlpha' , 0.3 , 'LineStyle' , 'none')
xlim([0,10])
ylim([-3 , 3])

plot(x , updated_samp1 , 'r--')
plot(x , updated_samp2 , 'g--')
plot(x , updated_samp3 , 'b--')

legend([TruePlot , TrainingPlot , PosteriorMeanPlot] , 'True Function' , 'Training Points' , 'Posterior Mean Function')
title('Posterior')
xlabel('x')
ylabel('y')



