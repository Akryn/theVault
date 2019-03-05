x = [0, 1, 0]';
y = [0, 0, 1]';
z = [0, 1, 2]';

plot3(x , y , z , 'kx')
grid on
hold on

%% triterp

qx = [];
qy = [];
qztriterp = [];
stepSize = 0.01;
for a = 0:stepSize:1
    b = 0;
    while b <= 1-a
        qx = [qx;a];
        qy = [qy;b];
        w = triterp(x,y,a,b);
        qztriterp = [qztriterp ; dot(w,z)];
        b = b+stepSize;
    end
end

plot3(qx,qy,qztriterp , 'r.')

%% scatteredInterpolant

F = scatteredInterpolant(x,y,z); % Default method is linear.
qz = F(qx,qy);
plot3(qx,qy,qz , 'bo')

logicalEqualWithError = qz<=qztriterp+eps & qz>=qztriterp-eps;
if sum(logicalEqualWithError) == length(qz)
    disp('Methods are equal to tolerance.')
else
    disp('Methods are NOT equal to tolerance.')
end


