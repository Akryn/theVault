%%% Neural Network Example Script

%% Load Data and Choose Cost and Activation Function.
load('mnist_all.mat')

% CF = @NN_CF_Quadratic; % Cost Function
grad_a__L_CF = @NN_CF_grad_a__L__Quadratic; % Gradient of the Cost Function with respect to a__l
AF = @NN_AF_Sigmoid; % Activation Function
AF_Prime = @NN_AF_Prime_Sigmoid; % Derivative of Activation Function;
z = @NN_AF_z__l; % Weighted input

% sigma = @( a__lm1 , w__l , b__l ) (AF(z( a__lm1 , w__l , b__l))); % AF(z)


%% Organising Data
X_Training = [train1 ; train2 ; train3 ; train4 ; train5 ; train6 ; train7 ; train8 ; train9 ; train0];

Y_Training = [repmat([1, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0] , size(train1 , 1) , 1) ;...
    repmat([0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0] , size(train2 , 1) , 1) ;...
    repmat([0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0] , size(train3 , 1) , 1) ;...
    repmat([0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0] , size(train4 , 1) , 1) ;...
    repmat([0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0] , size(train5 , 1) , 1) ;...
    repmat([0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0] , size(train6 , 1) , 1) ;...
    repmat([0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0] , size(train7 , 1) , 1) ;...
    repmat([0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0] , size(train8 , 1) , 1) ;...
    repmat([0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0] , size(train9 , 1) , 1) ;...
    repmat([0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1] , size(train0 , 1) , 1)];

% If we made our output to be the number the algorithm believes the image to be,
% our "error" no longer makes sense. For example, if we say our error is just the difference between
% the output and the actual value, 1-7 = -6 but 1-3 = -2. I would say 1 is closer to 7 than it is to
% 3. This also would not work since we use a sigmoid function and only output in the range [0,1].

% This leads us to having 10 Output neurons and a scale of [0,1] for each neuron. The algorithm will
% almost surely never get exactly a 1 in a single column and 0s in everything else, but it should
% pretty easy to make inferences after we receive the output neurons by using argmax. We cannot
% learn using argmax however due to the example above.

n_Training = length(Y_Training);

X_Test = [test1 ; test2 ; test3 ; test4 ; test5 ; test6 ; test7 ; test8 ; test9 ; test0];

Y_Test = [repmat([1, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0] , size(test1 , 1) , 1) ;...
    repmat([0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0] , size(test2 , 1) , 1) ;...
    repmat([0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0] , size(test3 , 1) , 1) ;...
    repmat([0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0] , size(test4 , 1) , 1) ;...
    repmat([0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0] , size(test5 , 1) , 1) ;...
    repmat([0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0] , size(test6 , 1) , 1) ;...
    repmat([0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0] , size(test7 , 1) , 1) ;...
    repmat([0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0] , size(test8 , 1) , 1) ;...
    repmat([0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0] , size(test9 , 1) , 1) ;...
    repmat([0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1] , size(test0 , 1) , 1)];

n_Test = length(Y_Test);

clearvars -except AF AF_Prime CF grad_a__L_CF z sigma n_Test n_Training X_Test X_Training Y_Test Y_Training

X_Training = double(X_Training);
Y_Training = double(Y_Training);
X_Test = double(X_Test);
Y_Test = double(Y_Test);
X_Training = X_Training./255; % Divide so our data goes from [0,255] to [0,1];
X_Test = X_Test./255;

% Should also split Training up into Training and Validation but that will come in the future.


%% Number of Neurons in each Layer (and hence the number of layers, including input and output) and Initialise Weights and Biases.
sizes = [784 , 30 , 10];
% Input layer has 784 neurons since the images of the handwriting are 28x28 pixels = 784 pixels.
% We have 10 output neurons since the digit can be 1 of 10 possibilities (0-9).
% We use 1 hidden layer with 30 neurons.
nL = length(sizes);

clear W B
W = cell(1 , nL-1);
B = cell(1 , nL-1);
% W and B have to be cell arrays since we do not require the same number of neurons in each layer.

for a= 1:(nL-1)
    W{a} = randn(sizes(a) , sizes(a+1)); % W{a}_jk is the weight between the j^th neuron in layer a and the k^th neuron in layer a+1.
    B{a} = randn(1 , sizes(a+1)); % B{a}_j is the bias to be added to the weighted sum going into the j^th neuron in layer a+1.
end

% Both Weights and Biases initialised randomly using samples from a Normal Distribution N(0,1). This can be improved on in the future.
% Note that the input layer doesn't have weights and biases so we only set those for nL-1 layers.


%% Set Mini-batch size (m), numer of epochs (nepochs) and Learning Rate (eta)
m = 10; % Mini-batch size (our random sample size for each pass through the learning algorithm).
nepochs = 1; % Number of epochs (how many times we use the full data set).
eta = 3; % Learning Rate
n = nepochs * n_Training;
nIter = n/m; % Total passes through algorithm.

RandPerms = [];
for a = 1:nepochs
    RandPerms = [RandPerms ; randperm(n_Training)];
end

X = [];
Y = [];
for a = 1:nepochs
    X = cat(4 , X , permute(reshape(X_Training(RandPerms(a,:) , :)',sizes(1),m,[]) , [2,1,3] ) ); % (Sample , RowImage , Mini-batch , Epoch)
    Y = cat(4 , Y , permute(reshape(Y_Training(RandPerms(a,:) , :)',sizes(end),m,[]) , [2,1,3] ) ); % (Sample , Label , Mini-batch , Epoch)
end

%% Training with Gradient Descent - THIS SECTION SHOULD BE TURNED INTO A FUNCTION
h=waitbar(0/nIter , ['Training via Gradient Descent - Mini-batch 1',' of ',num2str(nIter)],'Name','Training Neural Network');

for a = 1:nIter
    
    waitbar(a/nIter , h , ['Training via Gradient Descent - Mini-batch ',num2str(a),' of ',num2str(nIter)] )
    
    X_MB = X(:,:,a); % Loops through mini-batches. The 3rd index uses linear indexing and so goes through all epochs.
    Y_MB = Y(:,:,a);
    
    
    a__l = cell(m , length(sizes)-1);
    z__l = cell(m , length(sizes)-1);
    d__l = cell(m , length(B) - 1);
    
    for x = 1:m;
        %% Feed Forward
        [Temp_a , Temp_z] = NN_FF( X_MB(x,:) , W , B , z , AF);
        
        for b = 1:size(Temp_a,2);
            a__l{x,b}  = Temp_a{1,b};
            z__l{x,b}  = Temp_z{1,b};
        end
        
        clear Temp_a Temp_z
        
        
        %% Output Error d__l{ . , end} (d__L or dCbydz__L ) and Back Propogate Error d__l{ . , .}
        for c = length(B) : -1 : 1
            if c == length(B)
                d__l{x,c} =  NN_BP_1( grad_a__L_CF(a__l{x,end} , Y_MB(x,:) )  , AF_Prime(z__l{x,end}) );
                continue
            end
            d__l{x,c} = NN_BP_2( d__l{x,c+1} , AF_Prime(z__l{x,c}) , W{c+1} );
        end
        
        
    end
    
    %% Gradient Descent
    % Updating using gradC which is the average gradC_x for all training x takes a long time to compute.
    % Instead of averaging over all training x, approximate by averaging over just a mini-batch instead.
    % This means we update n_training / m times for a given epoch instead of once.
    
    for c = length(B): -1 : 1
        Temp = zeros( [size(W{c}) , m] );
        B{c} = B{c} - ((eta/m) .* sum( reshape([d__l{:,c}] , length(B{c}) , m)' ));
        
        if c == 1
            for x = 1:m
                Temp(:,:,x) = X_MB(x,:)' * d__l{x,c};
            end
            W{c} = W{c} - ((eta/m) .* sum(Temp , 3));
            continue
        end
        
        for x = 1:m
            Temp(:,:,x) = a__l{x,c-1}' * d__l{x,c};
        end
        W{c} = W{c} - ((eta/m) .* sum(Temp , 3));
    end
    clear Temp
    
end

close(h)

%% Test
X_Test_Loop = double(X_Test);
a__L = zeros(size(X_Test,1) , sizes(end));

h=waitbar(0/size(X_Test,1) , ['Testing Sample 1',' of ',num2str(size(X_Test,1))],'Name','Using Neural Network');
for a = 1 : size(X_Test,1);
    waitbar(a/size(X_Test,1) , h , ['Testing Sample ',num2str(a),' of ',num2str(size(X_Test,1))] )
    
    Temp_a = NN_FF( X_Test_Loop(a,:) , W , B , z , AF);
    a__L(a , :) = Temp_a{end};
end
close(h)

%% Decision

[~ , NumHat] = max(a__L , [] , 2);
NumHat(NumHat == 10) = 0;

[~ , Num] = max(Y_Test , [] , 2);
Num(Num == 10) = 0;

figure;
plot(1: size(X_Test,1) , NumHat , 'o')
hold on
plot(1:size(X_Test,1) , Num , '.')
title([num2str(100*sum(Num == NumHat)/size(X_Test,1)) , '%'])

%% Test (Training) - DISABLED

% X_Test_Loop = double(X_Training);
% a__L = zeros(size(X_Training,1) , sizes(end));
%
% h=waitbar(0/size(X_Training,1) , ['Testing Sample 1',' of ',num2str(size(X_Training,1))],'Name','Using Neural Network');
% for a = 1 : size(X_Training,1);
%     waitbar(a/size(X_Training,1) , h , ['Testing Sample ',num2str(a),' of ',num2str(size(X_Training,1))] )
%
%     Temp_a = NN_FF( X_Test_Loop(a,:) , W , B , z , AF);
%     a__L(a , :) = Temp_a{end};
% end
% close(h)

%% Decision (Training) - DISABLED

% [~ , NumHat] = max(a__L , [] , 2);
% NumHat(NumHat == 10) = 0;
%
% [~ , Num] = max(Y_Training , [] , 2);
% Num(Num == 10) = 0;
%
% figure;
% plot(1: size(X_Training,1) , NumHat , 'o')
% hold on
% plot(1:size(X_Training,1) , Num , '.')
% title([num2str(100*sum(Num == NumHat)/size(X_Training,1)) , '%'])
