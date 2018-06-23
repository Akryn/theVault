function LogicalBoundariesImage = IdentifyBoundaries(LogicalInputImage, Connectivity)
% Inputs:
% - LogicalInputImage = Image to find boundaries of where each element is either 0 or 1 (does not stricly have to be
%    logical). A logical image guarantees all elements are in the set [0,1].
% - Connectivity = Number to define what connectivity to use. Either 4 (up, down, left, right) or 8
%    (4 and diagonals).
% Outputs:
%  - LogicalBoundariesImage = Logical image which is true if the pixel is a boundary pixel and 0
%     otherwise.

[s1 , s2] = size(LogicalInputImage);

if sum(LogicalInputImage(:) == 0) + sum(LogicalInputImage(:) == 1) ~= s1 * s2
    error('Input image contains a number that is not in the set [0,1].')
end

if Connectivity ~= 4 && Connectivity ~= 8
    error('Specified connectivity is not in the set [4,8].')
end

%% Find the border

Up = [LogicalInputImage(2:end , :) ; zeros(1, s2)];
Down = [zeros(1, s2) ; LogicalInputImage(1:end-1 , :)];
Left = [LogicalInputImage(: , 2:end) , zeros(s1,1)];
Right = [zeros(s1,1) , LogicalInputImage(: , 1:end-1)];

if Connectivity == 8
    UpLeft = [Up(: , 2:end) , zeros(s1,1)];
    UpRight = [zeros(s1,1) , Up(: , 1:end-1)];
    DownLeft = [Down(: , 2:end) , zeros(s1,1)];
    DownRight = [zeros(s1,1) , Down(: , 1:end-1)];
end

if Connectivity == 4
    DecisionImage = Up + Down + Left + Right + LogicalInputImage;
    LogicalBoundariesImage = logical((DecisionImage ~= 5) .* LogicalInputImage);
else
    DecisionImage = Up + Down + Left + Right + UpLeft + UpRight + DownLeft + DownRight + LogicalInputImage;
    LogicalBoundariesImage = logical((DecisionImage ~= 9) .* LogicalInputImage);
end

end % End of function