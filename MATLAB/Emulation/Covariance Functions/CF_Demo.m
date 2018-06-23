s = 50;

[rr , cc] = ndgrid(1:s , 1:s);
X = cat(2 , rr(:) , cc(:)); clear rr cc

%% CF_Periodic - Varying L_p

KXX1 = CF_Periodic( X , X , 1 , [10,10]);

Line1 = reshape( KXX1(1,:) , s , s );
clear KXX1

KXX2 = CF_Periodic( X , X , 2 , [10,10]);

Line2 = reshape( KXX2(1,:) , s , s );
clear KXX2

KXX4 = CF_Periodic( X , X , 4 , [10,10]);

Line4 = reshape( KXX4(1,:) , s , s );
clear KXX4

KXX0p5 = CF_Periodic( X , X , 0.5 , [10,10]);

Line0p5 = reshape( KXX0p5(1,:) , s , s );
clear KXX0p5

KXX0p25 = CF_Periodic( X , X , 0.25 , [10,10]);

Line0p25 = reshape( KXX0p25(1,:) , s , s );
clear KXX0p25


figure;
P1 = plot(Line1(:,1));
hold on
P2 = plot(Line2(:,1));
P3 = plot(Line4(:,1));
P4 = plot(Line0p5(:,1));
P5 = plot(Line0p25(:,1));
xlabel('Pixel Distance')
ylabel('Correlation')
legend([P1 , P2 , P3 , P4 , P5] , 'L_p = 1' , 'L_p = 2' , 'L_p = 4' , 'L_p = 1/2' , 'L_p = 1/4')
title('Effect of L_p on CF\_Periodic with p = [10 , 10]')


figure;
imagesc(Line1);
title('CF\_Periodic with p = [10 , 10] and L_p = 1')
axis off

%% CF_LocallyPeriodic Varying L_p

KXX1L = CF_LocallyPeriodic( X , X , 1 , [10,10] , [10*4 , 10*4]);

Line1L = reshape( KXX1L(1,:) , s , s );
clear KXX1L

KXX2L = CF_LocallyPeriodic( X , X , 2 , [10,10], [10*4 , 10*4]);

Line2L = reshape( KXX2L(1,:) , s , s );
clear KXX2L

KXX4L = CF_LocallyPeriodic( X , X , 4 , [10,10] , [10*4 , 10*4]);

Line4L = reshape( KXX4L(1,:) , s , s );
clear KXX4L

KXX0p5L = CF_LocallyPeriodic( X , X , 0.5 , [10,10] , [10*4 , 10*4]);

Line0p5L = reshape( KXX0p5L(1,:) , s , s );
clear KXX0p5L

KXX0p25L = CF_LocallyPeriodic( X , X , 0.25 , [10,10] , [10*4 , 10*4]);

Line0p25L = reshape( KXX0p25L(1,:) , s , s );
clear KXX0p25L


figure;
P1 = plot(Line1L(:,1));
hold on
P2 = plot(Line2L(:,1));
P3 = plot(Line4L(:,1));
P4 = plot(Line0p5L(:,1));
P5 = plot(Line0p25L(:,1));
xlabel('Pixel Distance')
ylabel('Correlation')
legend([P1 , P2 , P3 , P4 , P5] , 'L_p = 1' , 'L_p = 2' , 'L_p = 4' , 'L_p = 1/2' , 'L_p = 1/4')
title('Effect of L_p on CF\_LocallyPeriodic with p = [10 , 10] and L = 4.*[10 , 10]')



KXX1L10_8 = CF_LocallyPeriodic( X , X , 1 , [10,10] , [10*8 , 10*8]);

Line1L10_8 = reshape( KXX1L10_8(1,:) , s , s );
clear KXX1L10_8

KXX1L10_4 = CF_LocallyPeriodic( X , X , 1 , [10,10] , [10*4 , 10*4]);

Line1L10_4 = reshape( KXX1L10_4(1,:) , s , s );
clear KXX1L10_4

KXX1L10_2 = CF_LocallyPeriodic( X , X , 1 , [10,10] , [10*2 , 10*2]);

Line1L10_2 = reshape( KXX1L10_2(1,:) , s , s );
clear KXX1L10_2

KXX1L10_1 = CF_LocallyPeriodic( X , X , 1 , [10,10] , [10*1 , 10*1]);

Line1L10_1 = reshape( KXX1L10_1(1,:) , s , s );
clear KXX1L10_1


figure;
P1 = plot(Line1L10_8(:,1));
hold on
P2 = plot(Line1L10_4(:,1));
P3 = plot(Line1L10_2(:,1));
P4 = plot(Line1L10_1(:,1));
xlabel('Pixel Distance')
ylabel('Correlation')
legend([P1 , P2 , P3 , P4] , 'L = 8.*[10 , 10]' , 'L = 4.*[10 , 10]' , 'L = 2.*[10 , 10]' , 'L = 1.*[10 , 10]')
title('Effect of L on CF\_LocallyPeriodic with p = [10 , 10] and L_p = 1')


figure;
imagesc(Line1L10_4);
title('CF\_LocallyPeriodic with p = [10 , 10], L_p = 1 and L = [40 , 40]')
axis off

%%

KXX1L_Single = CF_LocallyPeriodic( X , X , 1 , [10, inf] , [10*4 , 10*4]);

Line1L_Single = reshape( KXX1L_Single(1,:) , s , s );

figure;
imagesc(Line1L_Single);
title('CF\_LocallyPeriodic with p = [10 , inf], L_p = 1 and L = 40')
axis off