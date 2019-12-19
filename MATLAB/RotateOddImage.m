function rotatedO = myRotateFunctionOdd(O, theta)

% O = Object to be rotated.
% theta = angle in radians to be roated in an anticlockwise direction.

% rotatedO = rotated image.

%% 0. Convert to abs(theta) <= deg2rad(45)

theta = wrapToPi(theta); % Convert to [-pi, pi]

if ~(abs(theta) <= pi/4)
    signTheta = sign(theta);
    
    if signTheta == 1 % theta > deg2rad(45)
        
        if theta <= (3 * pi / 4) % deg2rad(45) < theta <= deg2rad(135)
            O = (fliplr(O))';
            theta = theta -(pi / 2);
        elseif theta <= pi % theta <= deg2rad(180)
            O = flipud(fliplr(O)); %#ok<*FLUDLR>
            theta = theta - pi;
        else
            error('theta > pi. This should not get hit and should be covered by theta = wrapToPi(theta).')
        end
        
    elseif signTheta == -1 % theta < deg2rad(-45)
        
        if theta >= -(3 * pi / 4) % deg2rad(-45) > theta >= deg2rad(-135)
            O = (flipud(O))';
            theta = theta + (pi / 2);
        elseif theta >= -pi % theta >= deg2rad(-180)
            O = flipud(fliplr(O));
            theta = theta + pi;
        else
            error('theta > pi. This should not get hit and should be covered by theta = wrapToPi(theta).')
        end
        
    else
       error('signTheta is neither 1 nor -1. This should not get hit and should be covered by theta == 0 case.') 
    end
    
end

if theta == 0 % No rotation required.
    rotatedO = O;
    return;
end

%% 1. Initial pad
N = size(O);
NNextPow2 = nextpow2(N);
NNextPow2Max = max(NNextPow2);
NNextPow2Max = [NNextPow2Max, NNextPow2Max];

N2 = 2.^(NNextPow2Max+1) + 1;
% N4 = 2.^(NNextPow2Max+2);

[OP] = padto(O, N2(1), N2(2), 0);

%% 2. Row FFT

FFTR = fft(OP, [], 2); % FFT on each row, i.e. over columns, hence dim = 2.

%% 3. Row Shear of tan(theta/2)

a = tan(theta/2);
delta_a = linspace(-a*floor(N2(1)/2), a*floor(N2(1)/2), N2(1))';
f_x =  [0:floor(N2(2)/2), -(fliplr(1:(ceil(N2(2)/2) - 1)))] ; % df = 1;

for row = 1:N2(1)
    % Delta causes exponential frequency to vary as a function of row i.e.,
    % causes shear. Normalise frequency by Nprime. This is similar to DSP
    % form of shift theorem, exp(1j*2*pi*delta*k/N). See:
    % https://ccrma.stanford.edu/~jos/mdft/Linear_Phase_Terms.html
    
    shiftFunc = exp(-1j.*2*pi.*f_x.*delta_a(row)./N2(2));
    FFTR_X(row,:) = FFTR(row,:).*shiftFunc;
end

%% 4. Row IFFT

OP_X = ifft(FFTR_X, [], 2);

%% 5. Column FFT

FFTC_X = fft(OP_X, [], 1);


%% 6. Column Shear of -sin(theta)

b = -sin(theta);
delta_b = linspace(-b*floor(N2(2)/2), b*floor(N2(2)/2), N2(2));
f_y =  [0:floor(N2(1)/2), -(fliplr(1:(ceil(N2(1)/2) - 1)))]' ; % df = 1;

for col = 1:N2(2)
    shiftFunc = exp(-1j.*2*pi.*f_y.*delta_b(col)./N2(1));
    FFTC_XY(:,col) = FFTC_X(:,col).*shiftFunc;
end

%% 7. Column IFFT

OP_XY = ifft(FFTC_XY, [], 1);

%% 8. Row FFT

FFTR_XY = fft(OP_XY, [], 2);

%% 9. Row Shear of tan(theta/2)

for row = 1:N2(1)
    shiftFunc = exp(-1j.*2*pi.*f_x.*delta_a(row)./N2(2));
    FFTR_XYX(row,:) = FFTR_XY(row,:).*shiftFunc;
end

%% 10. Row IFFT

OP_XYX = ifft(FFTR_XYX, [], 2);

%% 11. Real

rotatedO = real(OP_XYX);
N2Cent =  floor(N2/2) + 1;
NWidth =  floor(N/2);
rotatedO = rotatedO(N2Cent(1) - NWidth(1) : N2Cent(1) + NWidth(1), N2Cent(2) - NWidth(2) : N2Cent(2) + NWidth(2));

end % End of function