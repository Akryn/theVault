function [MaxDistance, CoordinatesOfEndPoints] = CalculateMaxDistance(LogicalInputImage)
% Inputs:
% - LogicalInputImage = Image to find the max distance between all elements equal to 1.
%    Each element is either 0 or 1 (does not stricly have to be logical).
%    A logical image guarantees all elements are in the set [0,1].
% Outputs:
% - MaxDistance = The max distance between all elements equal to 1.
% - CoordinatesOfEndPoints = (n x 4) matrix where n is the number of pairs of points which have a
%    distance between them equal to MaxDistance.
%    [point1_row, point1_column, point2_row, point2_column]
%    Unsorted.
%
% TIP: This function may run faster if you run CalculateMaxDistance(IdentifyBoundaries(LogicalInputImage,4)).
%  Since the maximum distance will always be between boundary points, only considering boundary
%  points may give us a speed improvement if the cost of identifying the boundaries is smaller than
%  the cost of also considering non-boundary points inside this function.

%% Checking inputs

[s1 , s2] = size(LogicalInputImage);

if sum(LogicalInputImage(:) == 0) + sum(LogicalInputImage(:) == 1) ~= s1 * s2
    error('Input image contains a number that is not in the set [0,1].')
end

%% Calculating max distance

[r , c] = find(LogicalInputImage);
[r1 , r2] = ndgrid(r , r);
[c1 , c2] = ndgrid(c , c);

SquaredDistances = (r2 - r1 ).^2 + (c2 - c1).^2; % Squared Distances - actually calculated them each twice, A to B and B to A.
MaxSquaredDistance = max(SquaredDistances(:));
MaxDistance = sqrt(MaxSquaredDistance);
if nargout > 1
    MaxIndices = find(SquaredDistances == MaxSquaredDistance);
    MaxIndices = MaxIndices(1 : (length(MaxIndices)/2) ); % Removing duplicates.
    CoordinatesOfEndPoints = [r1(MaxIndices) , c1(MaxIndices) , r2(MaxIndices) , c2(MaxIndices)];
end

end % End of function