function w = triterp(x,y,qx,qy)
% Barycentric Coordinates:
% https://codeplea.com/triangular-interpolation
    denom  = ( (y(2) - y(3)) * (x(1) - x(3)) ) + ( (x(3) - x(2)) * (y(1) - y(3)) );
    numer1 = ( (y(2) - y(3)) * ( qx  - x(3)) ) + ( (x(3) - x(2)) * ( qy  - y(3)) );
    numer2 = ( (y(3) - y(1)) * ( qx  - x(3)) ) + ( (x(1) - x(3)) * ( qy  - y(3)) );
    w1 = numer1 / denom;
    w2 = numer2 / denom;
    w3 = 1 - w1 - w2;
    w = [w1, w2, w3];
end

