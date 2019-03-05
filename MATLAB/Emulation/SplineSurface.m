[s1, s2] = size(LogicalGrid);
[xq,yq] = ndgrid(1:s1, 1:s2);
m = griddata(X(:,1),X(:,2),Y,xq,yq, 'v4'); 
% Biharmonic spline interpolation supporting 2-D interpolation only. Not based on triangulation
% Averages z values where we have multiple training points at a single input vector.
% Determines distances between training points.
% Determine weights for interpolation via Green's function
m = m(LogicalGrid);


figure;
TrainingPlot = plot3(X(:,1) , X(:,2) , Y , 'bx');
hold on;
EmulationPlot = plot3(Xstar(:,1) , Xstar(:,2) , m , 'r.');
grid on
legend([TrainingPlot , EmulationPlot] , 'Training' , 'Emulated' , 'Location' , 'southoutside')
