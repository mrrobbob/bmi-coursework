function [modelParameters] = positionEstimatorTraining(training_data)
% Get firing rates
% Sampling frequency is 1000 Hz?
trainingData = struct([]);
spike_rate = [];
firingrates = [];


% velocities = struct([]);

% bin_size = 14;
% fs = 1000;
max_t_length=zeros(1,8);
v_predict= struct([]);
for k = 1:8
    max_t=0;
    for n = 1:length(training_data)
        max_t= max([length(training_data(n,k).handPos), max_t]);
    end
    max_t_length(k)= max_t;
end

for k = 1:8
    v_x = zeros(1, max_t_length(k));
    v_y = zeros(1, max_t_length(k));
    for n= 1:length(training_data)
        if max_t_length(k)<= length(training_data(n,k).handPos(1, 320:end))
            v_x= v_x+training_data(n,k).handPos(1, 320:max_t_length(k));
            v_y= v_y+training_data(n,k).handPos(2, 320:max_t_length(k));
        else
            temp_x= ones(1, max_t_length(k)- length(training_data(n,k).handPos(1,320:end)))*training_data(n,k).handPos(1, end);
            temp_y= ones(1, max_t_length(k)- length(training_data(n,k).handPos(1,320:end)))*training_data(n,k).handPos(2, end);
            v_x= v_x+[training_data(n,k).handPos(1,320:end) temp_x];
            v_y= v_y+[training_data(n,k).handPos(2,320:end) temp_y];
        end


    end
    v_predict(k).meanX= v_x/length(training_data);
    v_predict(k).meanY= v_y/length(training_data);
end

spikes = [];
reachingAngle = [];
spikeCount = zeros(length(training_data),98);

for k = 1:8
    for i = 1:98
        for n = 1:length(training_data)
                number_of_spikes =sum(training_data(n,k).spikes(i,1:320));
                spikeCount(n,i) = number_of_spikes;
        end
    end
    spikes = cat(1, spikes, spikeCount);
    reaching_angle(1:length(training_data)) = k;
    reachingAngle = cat(2, reachingAngle, reaching_angle);
end

% Neural network to predict angle (classification)
layer_sizes = [98, 70, 8];
maxIter = 500;
lambda = 0.6;
alpha = 0.4;
% loss = [];
% count = [];
% precisionT = [];
%m = size(spikes,1);


for i = 1:(size(layer_sizes, 2)-1)
    initial_weights{i} = initweights(layer_sizes(i), layer_sizes(i+1));
end

weights = initial_weights;

% figure(1);
% ax1 = subplot(2,1,1); ax1.YGrid = "on"; ax1.XGrid = "on";
% ax2 = subplot(2,1,2); ax2.YGrid = "on"; ax2.XGrid = "on";

reachingAnglenew = (1:8) == reachingAngle';

% tic; 
for i = 1:maxIter
    [~, weights] = train(weights, layer_sizes,spikes, reachingAnglenew, lambda, alpha);
end

modelParameters = struct('v_predict',v_predict,'nnweights',weights);

end

% Functions for the neural network.

function W = initweights(L_in, L_out)
%W = zeros(L_out, 1 + L_in);
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
end

function p = nnpredict(weights, X, layer_sizes)
m = size(X, 1);
%p = zeros(size(X, 1), 1);

% Init sum and activation.
for i = 1:(size(layer_sizes,2))
    z{i} = zeros(m,layer_sizes(i));
    a{i} = zeros(m,layer_sizes(i));
end

a{1,1} = [ones(m, 1) X]; 

for i = 2:(size(layer_sizes,2))
    z{1,i} = a{1,i-1}*weights{1,i-1}';
    a{1,i} = sigmoid(z{1,i});
    if i ~= size(layer_sizes,2)
        a{1,i} = [ones(m,1) a{1,i}];
    end
end

[~, p] = max(a{1,end}, [], 2);
end

function x = sigmoid(z)
    x = 1.0 ./ (1.0 + exp(-z));
end

function x = sigmoidGradient(z)
s = sigmoid(z);
x = s.*(1-s);
end

function [J, newWeights] = train(weights, layer_sizes,X, y_new, lambda, alpha)
    % Useful variables
    m = size(X, 1);
    a1 = [ones(m, 1) X];   
    %J = 0;

    % Init sum and activation.
    for i = 1:(size(layer_sizes,2))
        z{i} = zeros(m,layer_sizes(i)+1);
        a{i} = zeros(m,layer_sizes(i)+1);
    end

    a{1,1} = a1;  

    % Forward propagation
    for i = 2:size(layer_sizes,2)
        z{1,i} = a{1,i-1}*weights{1,i-1}';
        a{1,i} = sigmoid(z{1,i});
        if i ~= size(layer_sizes,2)
            a{1,i} = [ones(length(z{1,i}),1) a{1,i}];
        end
    end

    % Cost function
    J = sum(sum(-y_new .* log(a{1,end}) - (1 - y_new) .* log(1 - a{1,end}))) / m ;

    % Regularisation of cost function
    regularisation = 0;
    for i = 1:(size(layer_sizes,2)-1)
        regularisation = regularisation + sum(weights{1,i}(:,2:end).^2,'all');
    end

    regularisation =  lambda/(2*m) * regularisation;
    J = J + regularisation;

    % Back propagation 

    for i = size(layer_sizes,2):-1:2
        if i == size(layer_sizes,2)
            s{i} = a{1,i} - y_new;
        else
            s{i} = (s{1,i+1}*weights{1,i}(:, 2:end)) .* sigmoidGradient(z{1,i});
        end        
    end

    for i = 1:(size(layer_sizes,2)-1)
        deltas{i} = s{1,i+1}'*a{1,i};
    end

    % Gradient descent
    for i = 1:(size(layer_sizes,2)-1) 
        p = lambda * [zeros(size(weights{1,i}, 1), 1), weights{1,i}(:, 2:end)] / m;
        weight_grad{i} = deltas{1,i}./m + p;

        newWeights{i} = zeros(size(weights{1,i}));
        
        newWeights{1,i}(:,1) = weights{1,i}(:,1) - alpha*mean(weight_grad{1,i}(:,1),1);
        newWeights{1,i}(:,2:end) = weights{1,i}(:,2:end) - alpha*weight_grad{1,i}(:,2:end);   
    end
end

% rr = reachingAngle';
% i = randi(length(reachingAngle'));
% pred = nnpredict(weights, spikes(i,:), layer_sizes);
% fprintf('True class: %d  |  Predicted class: %d\n',rr(i),pred);