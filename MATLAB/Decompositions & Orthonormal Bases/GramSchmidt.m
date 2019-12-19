% Regression is more numerically stable when the condition number of the design matrix is smaller.
% One way to reduce the condition number is by transforming the columns to be orthogonal to each
% other.

% NOTE: https://en.wikipedia.org/wiki/QR_decomposition#Advantages_and_disadvantages 
% "The Gram-Schmidt process is inherently numerically unstable. While the application of the
% projections has an appealing geometric analogy to orthogonalization, the orthogonalization itself
% is prone to numerical error. A significant advantage however is the ease of implementation, which
% makes this a useful algorithm to use for prototyping if a pre-built linear algebra library is
% unavailable."

% In the future, I should investigate:
% Householder Reflections which is harder to implement but more numerically stable than the
% Gram-Schmidt process, bandwidth heavy and not parallelisable.
% Givens Rotations which is harder to implement than Householder Reflections but more bandwidth
% efficient and more parralelisable.
% https://en.wikipedia.org/wiki/QR_decomposition#Computing_the_QR_decomposition for methods.

%% Implementing Gram-Schmidt from Example
% Based on https://en.wikipedia.org/wiki/QR_decomposition#Example
A = [12, -51, 4;
    6, 167, -68;
    -4, 24, -41];

u1 = A(:,1);
u2 = A(:,2) - ((u1' * A(:,2)) / (u1' * u1)) .* u1;
u3 = A(:,3) - ((u1' * A(:,3)) / (u1' * u1)) .* u1 - ((u2' * A(:,3)) / (u2' * u2)) .* u2;

U = [u1, u2, u3];

Q = U ./ sqrt(sum(U.^2));

R = Q' * A;

%% Using MATLAB's QR Decomposition Function
A = [12, -51, 4;
    6, 167, -68;
    -4, 24, -41];

[Qmat, Rmat] = qr(A);

% Q and R are very close to Qmat and Rmat.
% Qmat and Rmat appear to be better.

%% Implementing Gram-Schmidt Myself
% Based on https://www.khanacademy.org/math/linear-algebra/alternate-bases/orthonormal-basis/v/linear-algebra-the-gram-schmidt-process
% u_i are now unit vectors and U the matrix of unit vectors.
% V is the matrix of vectors to be orthonormalised.

V = [12, -51, 4;
    6, 167, -68;
    -4, 24, -41];

figure;
quiver3(0, 0, 0, V(1,1), V(2,1), V(3,1), 'r', 'DisplayName', 'v1', 'AutoScale', 'off')
hold on
quiver3(0, 0, 0, V(1,2), V(2,2), V(3,2), 'b', 'DisplayName', 'v2', 'AutoScale', 'off')
quiver3(0, 0, 0, V(1,3), V(2,3), V(3,3), 'g', 'DisplayName', 'v3', 'AutoScale', 'off')
maxV = max(V(:));
xlim([-maxV, maxV])
ylim([-maxV, maxV])
zlim([-maxV, maxV])
xlabel('x')
ylabel('y')
zlabel('z')
legend('Location', 'best')

%
v1 = V(:,1);
% We could calculate w1 here but it would be the 0 vector.
u1 = v1 ./ sqrt(v1' * v1); % The first unit vector. It spans a line.

figure;
subplot(1,2,1)
quiver3(0, 0, 0, v1(1), v1(2), v1(3), 'r', 'DisplayName', 'v1', 'AutoScale', 'off')
max1 = max(v1);
xlim([-max1, max1])
ylim([-max1, max1])
zlim([-max1, max1])
xlabel('x')
ylabel('y')
zlabel('z')
legend('Location', 'best')

subplot(1,2,2)
quiver3(0, 0, 0, u1(1), u1(2), u1(3), 'r', 'DisplayName', 'u1', 'AutoScale', 'off')
xlim([-1,1])
ylim([-1,1])
zlim([-1,1])
xlabel('x')
ylabel('y')
zlabel('z')
legend('Location', 'best')

%
v2 = V(:,2);
v2proju1 = (v2' * u1).*u1; % Calculate the projection of v2 on to the line spanned by u1.
w2 = v2 - v2proju1; % Subtract the projection to make an orthogonal vector.
u2 = w2 ./ sqrt(w2' * w2); % Normalise.

figure;
subplot(1,2,1)
quiver3(0, 0, 0, u1(1), u1(2), u1(3), 'r', 'DisplayName', 'u1', 'AutoScale', 'off')
hold on
quiver3(0, 0, 0, v2(1), v2(2), v2(3), 'b', 'DisplayName', 'v2', 'AutoScale', 'off')
quiver3(0, 0, 0, v2proju1(1), v2proju1(2), v2proju1(3), 'm', 'DisplayName', 'v2proju1', 'AutoScale', 'off')
quiver3(v2proju1(1), v2proju1(2), v2proju1(3), w2(1), w2(2), w2(3), 'b--', 'DisplayName', 'w2', 'AutoScale', 'off')
max12 = max(abs([u1;v2;v2proju1]));
xlim([-max12, max12])
ylim([-max12, max12])
zlim([-max12, max12])
xlabel('x')
ylabel('y')
zlabel('z')
legend('Location', 'best')

subplot(1,2,2)
quiver3(0, 0, 0, u1(1), u1(2), u1(3), 'r', 'DisplayName', 'u1', 'AutoScale', 'off')
hold on
quiver3(0, 0, 0, u2(1), u2(2), u2(3), 'b', 'DisplayName', 'u2', 'AutoScale', 'off')
xlim([-1,1])
ylim([-1,1])
zlim([-1,1])
xlabel('x')
ylabel('y')
zlabel('z')
legend('Location', 'best')

%
v3 = V(:,3);
v3proju1u2 = (v3' * u1).*u1 + (v3' * u2).*u2; % Calculate the projection of v3 on to the surface spanned by u1 and u2.
w3 = v3 - v3proju1u2; % Subtract the projection to make an orthogonal vector.
u3 = w3 ./ sqrt(w3' * w3); % Normalise.

figure;
subplot(1,2,1)
quiver3(0, 0, 0, u1(1), u1(2), u1(3), 'r', 'DisplayName', 'u1', 'AutoScale', 'off')
hold on
quiver3(0, 0, 0, u2(1), u2(2), u2(3), 'b', 'DisplayName', 'u2', 'AutoScale', 'off')
quiver3(0, 0, 0, v3(1), v3(2), v3(3), 'g', 'DisplayName', 'v3', 'AutoScale', 'off')
quiver3(0, 0, 0, v3proju1u2(1), v3proju1u2(2), v3proju1u2(3), 'c', 'DisplayName', 'v2proju1u2', 'AutoScale', 'off')
quiver3(v3proju1u2(1), v3proju1u2(2), v3proju1u2(3), w3(1), w3(2), w3(3), 'g--', 'DisplayName', 'w3', 'AutoScale', 'off')
max123 = max(abs([u1;u2;v3;v3proju1u2]));
xlim([-max123, max123])
ylim([-max123, max123])
zlim([-max123, max123])
xlabel('x')
ylabel('y')
zlabel('z')
legend('Location', 'best')

subplot(1,2,2)
quiver3(0, 0, 0, u1(1), u1(2), u1(3), 'r', 'DisplayName', 'u1', 'AutoScale', 'off')
hold on
quiver3(0, 0, 0, u2(1), u2(2), u2(3), 'b', 'DisplayName', 'u2', 'AutoScale', 'off')
quiver3(0, 0, 0, u3(1), u3(2), u3(3), 'g', 'DisplayName', 'u3', 'AutoScale', 'off')
xlim([-1,1])
ylim([-1,1])
zlim([-1,1])
xlabel('x')
ylabel('y')
zlabel('z')
legend('Location', 'best')

Qmy = [u1, u2, u3];

% Because Qmy is an orthogonal matrix, Q' = inv(Q);
% Therefore to solve for R in V = QR, R = inv(Q) * V = Q' * V
Rmy = Qmy' * V;

%% Implementing Modified Gram-Schmidt (MGS)
% https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Numerical_stability:

% "When this process is implemented on a computer, the vectors {\displaystyle \mathbf {u} _{k}}
% \mathbf {u} _{k} are often not quite orthogonal, due to rounding errors. For the Gram–Schmidt
% process as described above (sometimes referred to as "classical Gram–Schmidt") this loss of
% orthogonality is particularly bad; therefore, it is said that the (classical) Gram–Schmidt process
% is numerically unstable. The Gram–Schmidt process can be stabilized by a small modification; this
% version is sometimes referred to as modified Gram-Schmidt or MGS. This approach gives the same
% result as the original formula in exact arithmetic and introduces smaller errors in
% finite-precision arithmetic."

% Instead of calculating the projection with the previous hypersurface and subtracting, we
% sequentially find the projection with a vector, subtract and repeat for all remaining vectors.
% In a 3D example, the difference is only visible for the 3rd vector.


V = [12, -51, 4;
    6, 167, -68;
    -4, 24, -41];

figure;
quiver3(0, 0, 0, V(1,1), V(2,1), V(3,1), 'r', 'DisplayName', 'v1', 'AutoScale', 'off')
hold on
quiver3(0, 0, 0, V(1,2), V(2,2), V(3,2), 'b', 'DisplayName', 'v2', 'AutoScale', 'off')
quiver3(0, 0, 0, V(1,3), V(2,3), V(3,3), 'g', 'DisplayName', 'v3', 'AutoScale', 'off')
maxV = max(V(:));
xlim([-maxV, maxV])
ylim([-maxV, maxV])
zlim([-maxV, maxV])
xlabel('x')
ylabel('y')
zlabel('z')
legend('Location', 'best')

%
v1 = V(:,1);
% We could calculate w1 here but it would be the 0 vector.
u1 = v1 ./ sqrt(v1' * v1); % The first unit vector. It spans a line.

figure;
subplot(1,2,1)
quiver3(0, 0, 0, v1(1), v1(2), v1(3), 'r', 'DisplayName', 'v1', 'AutoScale', 'off')
max1 = max(v1);
xlim([-max1, max1])
ylim([-max1, max1])
zlim([-max1, max1])
xlabel('x')
ylabel('y')
zlabel('z')
legend('Location', 'best')

subplot(1,2,2)
quiver3(0, 0, 0, u1(1), u1(2), u1(3), 'r', 'DisplayName', 'u1', 'AutoScale', 'off')
xlim([-1,1])
ylim([-1,1])
zlim([-1,1])
xlabel('x')
ylabel('y')
zlabel('z')
legend('Location', 'best')

%
v2 = V(:,2);
v2proju1 = (v2' * u1).*u1; % Calculate the projection of v2 on to the line spanned by u1.
w2 = v2 - v2proju1; % Subtract the projection to make an orthogonal vector.
u2 = w2 ./ sqrt(w2' * w2); % Normalise.

figure;
subplot(1,2,1)
quiver3(0, 0, 0, u1(1), u1(2), u1(3), 'r', 'DisplayName', 'u1', 'AutoScale', 'off')
hold on
quiver3(0, 0, 0, v2(1), v2(2), v2(3), 'b', 'DisplayName', 'v2', 'AutoScale', 'off')
quiver3(0, 0, 0, v2proju1(1), v2proju1(2), v2proju1(3), 'm', 'DisplayName', 'v2proju1', 'AutoScale', 'off')
quiver3(v2proju1(1), v2proju1(2), v2proju1(3), w2(1), w2(2), w2(3), 'b--', 'DisplayName', 'w2', 'AutoScale', 'off')
max12 = max(abs([u1;v2;v2proju1]));
xlim([-max12, max12])
ylim([-max12, max12])
zlim([-max12, max12])
xlabel('x')
ylabel('y')
zlabel('z')
legend('Location', 'best')

subplot(1,2,2)
quiver3(0, 0, 0, u1(1), u1(2), u1(3), 'r', 'DisplayName', 'u1', 'AutoScale', 'off')
hold on
quiver3(0, 0, 0, u2(1), u2(2), u2(3), 'b', 'DisplayName', 'u2', 'AutoScale', 'off')
xlim([-1,1])
ylim([-1,1])
zlim([-1,1])
xlabel('x')
ylabel('y')
zlabel('z')
legend('Location', 'best')

%
v3 = V(:,3);
v3proju1 = (v3' * u1).*u1; % Calculate the projection of v3 on to the line spanned by u1.
w3_1 = v3 - v3proju1; % Subtract the projection to make an orthogonal vector to u1.
w3_1proju2 = (w3_1' * u2).*u2; % Calculate the projection of w3_1 on to the line spanned by u2.
w3_2 = w3_1 - w3_1proju2; % Subtract the projection to make an orthogonal vector to u2.
u3 = w3_2 ./ sqrt(w3_2' * w3_2); % Normalise.

figure;
subplot(1,2,1)
quiver3(0, 0, 0, u1(1), u1(2), u1(3), 'r', 'DisplayName', 'u1', 'AutoScale', 'off')
hold on
quiver3(0, 0, 0, u2(1), u2(2), u2(3), 'b', 'DisplayName', 'u2', 'AutoScale', 'off')
quiver3(0, 0, 0, v3(1), v3(2), v3(3), 'g', 'DisplayName', 'v3', 'AutoScale', 'off')
quiver3(0, 0, 0, v3proju1(1), v3proju1(2), v3proju1(3), 'm', 'DisplayName', 'v3proju1', 'AutoScale', 'off')
quiver3(v3proju1(1), v3proju1(2), v3proju1(3), w3_1(1), w3_1(2), w3_1(3), 'g--', 'DisplayName', 'w3_1', 'AutoScale', 'off')
quiver3(0, 0, 0, w3_1(1), w3_1(2), w3_1(3), 'Color', [0,0.5,0], 'DisplayName', 'w3_1', 'AutoScale', 'off')
quiver3(0, 0, 0, w3_1proju2(1), w3_1proju2(2), w3_1proju2(3), 'c', 'DisplayName', 'w3_1proju2', 'AutoScale', 'off')
quiver3(w3_1proju2(1), w3_1proju2(2), w3_1proju2(3), w3_2(1), w3_2(2), w3_2(3), '--', 'Color', [0,0.5,0], 'DisplayName', 'w3_2', 'AutoScale', 'off')
max123 = max(abs([u1;u2;v3;v3proju1;w3_1proju2]));
xlim([-max123, max123])
ylim([-max123, max123])
zlim([-max123, max123])
xlabel('x')
ylabel('y')
zlabel('z')
legend('Location', 'best')

subplot(1,2,2)
quiver3(0, 0, 0, u1(1), u1(2), u1(3), 'r', 'DisplayName', 'u1', 'AutoScale', 'off')
hold on
quiver3(0, 0, 0, u2(1), u2(2), u2(3), 'b', 'DisplayName', 'u2', 'AutoScale', 'off')
quiver3(0, 0, 0, u3(1), u3(2), u3(3), 'g', 'DisplayName', 'u3', 'AutoScale', 'off')
xlim([-1,1])
ylim([-1,1])
zlim([-1,1])
xlabel('x')
ylabel('y')
zlabel('z')
legend('Location', 'best')

QmyMod = [u1, u2, u3];

% Because Qmy is an orthogonal matrix, Q' = inv(Q);
% Therefore to solve for R in V = QR, R = inv(Q) * V = Q' * V
RmyMod = QmyMod' * V;
