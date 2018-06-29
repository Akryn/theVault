x = [0:0.1:10]';
y = sin(x);

x_train = [2 ; 5 ; 7 ; 9];
y_train = sin(x_train);

nugget = 0.000001;

KXX = CF_SquaredExponential(x_train,x_train, 1);
KXX = KXX + (nugget .* eye(length(x_train)));

KXstarX = CF_SquaredExponential(x , x_train, 1);

f_star = KXstarX * (KXX  \ y_train);

KXstarXInvKXX = KXstarX / KXX;
KXstarXInvKXXRowSums = sum(KXstarXInvKXX , 2);

figure;
plot(x , KXstarXInvKXX(:,1))
hold on
plot(x , KXstarXInvKXX(:,2))
plot(x , KXstarXInvKXX(:,3))
plot(x , KXstarXInvKXX(:,4))
title('Weighting of Training point y_i when predicting at test x_*')
ylabel('Weighting')
xlabel('x_*')
legend('y_1 @ x_1 = 2' , 'y_2 @ x_2 = 5', 'y_3 @ x_3 = 7', 'y_4 @ x_4 = 9' , 'location' ,'best')

