% Recreating R's poly() function.
% Adapted from
% https://stackoverflow.com/questions/39031172/how-poly-generates-orthogonal-polynomials-how-to-understand-the-coefs-ret/39051154#39051154
% https://stats.stackexchange.com/questions/253123/what-are-multivariate-orthogonal-polynomials-as-computed-in-r
% is also a good resource.

%% Single Covariate
% [1, x, x.^2, x.^3]

clear
x = (0:20)';

[Q, ~, orthonormPredict] = Rpoly(x, 3);

xstar = (0:0.5:20)';
Qstar = Rpoly_predict(xstar, orthonormPredict);
% Correlations appear between Qstar(:,2) (linear) and Qstar(:,4) (cubic) which didn't exist in Q.
% This also occurs in R,

figure;
for a = 1 : size(Q,2)
    if a == 1
        plot(x, Q(:,a), 'b-+', 'DisplayName', 'Q')
        hold on
        grid on
        xlabel('x')
        ylabel('Z')
        legend('Location', 'best')
        title("Demonstration of R's poly()")
        plot(xstar, Qstar(:,a), 'r--x', 'DisplayName', 'Qstar')
    else
        plot(x, Q(:,a), 'b-+', 'HandleVisibility','off')
        plot(xstar, Qstar(:,a), 'r--x', 'HandleVisibility', 'off')
    end
end

%% Multiple Covariates
% [1, x(:,1), x(:,1).^2, x(:,1).^3, x(:,2), x(:,1).*x(:,2), x(:,2).^2]

% I originally had x [(1:20)', (1:20)'] and was having the problem that my columns of my design
% matrix were highly correlated even after using poly() in R. However, this makes sense because
% there is no way to tell which covariate/predictor to attribute changes in y to if the
% covariate/predictors are identical. I effectively had constrained my training points to the line y
% = x. What should be (and is typically) done is to use ndgrid(). This way we can see how changed
% one covariate/predictor affects y when we change it while keeping others fixed. In this case I
% will have a grid in the x-y space.

clear
xrVec = (0:20)';
xcVec = (10:40)';
[xr, xc] = ndgrid(xrVec, xcVec);
x = [xr(:), xc(:)]; clear xx xr xc
% x = single(x);

[Q1, ~, orthonormPredict1] = Rpoly(x(:,1), 3);
[Q2, ~, orthonormPredict2] = Rpoly(x(:,2), 2);
Q = [Q1, Q2(:,2), Q1(:,2) .* Q2(:,2), Q2(:,3)];

xrVecstar = (0:0.5:20)';
xcVecstar = (10:0.5:40)';
[xrstar, xcstar] = ndgrid(xrVecstar, xcVecstar);
xstar = [xrstar(:), xcstar(:)]; clear xstarxstar xrstar xcstar

Qstar1 = Rpoly_predict(xstar(:,1), orthonormPredict1);
Qstar2 = Rpoly_predict(xstar(:,2), orthonormPredict2);
Qstar = [Qstar1, Qstar2(:,2), Qstar1(:,2) .* Qstar2(:,2), Qstar2(:,3)];
% Correlations appear between Qstar(:,2) (linear) and Qstar(:,4) (cubic) which didn't exist in Q.
% This also occurs in R,

figure;
for a = 1 : size(Q,2)
    if a == 1
        plot3(x(:,1), x(:,2), Q(:,a), 'bx', 'DisplayName', 'Q')
        hold on
        grid on
        xlabel('x')
        ylabel('Z')
        legend('Location', 'best')
        title("Demonstration of R's poly()")
        plot3(xstar(:,1), xstar(:,2), Qstar(:,a), 'r.', 'DisplayName', 'Qstar')
    else
        plot3(x(:,1), x(:,2), Q(:,a), 'bx', 'HandleVisibility','off')
        plot3(xstar(:,1), xstar(:,2), Qstar(:,a), 'r.', 'HandleVisibility', 'off')
    end
end


%% Single Covariate - Non-Polynomial
% [1, x, x.^2, exp(x)]

clear
x = (0:20)';

[Q1, ~, orthonormPredict1] = Rpoly(x, 2);
[Q2, ~, orthonormPredict2] = Rpoly(exp(x), 1);
Q = [Q1, Q2(:,2)];
% Correlations exist between Q(:,2) (linear) and Q(:,4).

xstar = (0:0.5:20)';
Qstar1 = Rpoly_predict(xstar, orthonormPredict1);
Qstar2 = Rpoly_predict(exp(xstar), orthonormPredict2);
Qstar = [Qstar1, Qstar2(:,2)];
% The correlations that were in Q still exist in Qstar.

figure;
for a = 1 : size(Q,2)
    if a == 1
        plot(x, Q(:,a), 'b-+', 'DisplayName', 'Q')
        hold on
        grid on
        xlabel('x')
        ylabel('Z')
        legend('Location', 'best')
        title("Demonstration of R's poly()")
        plot(xstar, Qstar(:,a), 'r--x', 'DisplayName', 'Qstar')
    else
        plot(x, Q(:,a), 'b-+', 'HandleVisibility','off')
        plot(xstar, Qstar(:,a), 'r--x', 'HandleVisibility', 'off')
    end
end