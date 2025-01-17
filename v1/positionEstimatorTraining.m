%load("monkeydata_training.mat");
function [modelParameters] = positionEstimatorTraining(training_data)
% Get firing rates
% Sampling frequency is 1000 Hz?
tic;
%training_data = trial;
trainingData = struct([]);
spike_rate = [];
firingrates = [];
v_x = [];
v_y = [];

velocities = struct([]);

bin_size = 14; %14
% Sampling frequency
fs = 1000; %1000

for k = 1:8
    for i = 1:98
        for n = 1:length(training_data)
            for j = 320:bin_size:570-bin_size
                % Get firing rates of one neuron unit for each trial
                num_spikes = length(find(training_data(n,k).spikes(i,j:j+bin_size)==1));
                spike_rate = cat(2, spike_rate, num_spikes * fs / bin_size);

                % Get velocity of moving hand
                if i==1
                    x_v_ = (training_data(n,k).handPos(1,j+bin_size) - training_data(n,k).handPos(1,j)) * fs / bin_size;
                    y_v_ = (training_data(n,k).handPos(2,j+bin_size) - training_data(n,k).handPos(2,j)) * fs / bin_size;
                    v_x = cat(2, v_x, x_v_);
                    v_y = cat(2, v_y, y_v_);
                end
            end
            firingrates = cat(2, firingrates, spike_rate);
            spike_rate = [];
        end
        trainingData(i,k).firingRates = firingrates;
        velocities(k).x = v_x;
        velocities(k).y = v_y;

        firingrates = [];
    end
    v_x = [];
    v_y = [];
end

% Linear Regression to predict velocity. (experiment on changing this)

v_predict = struct([]);

for k=1:8
    vel = [velocities(k).x; velocities(k).y];
    firingRate = [];
    for i=1:98
        firingRate = cat(1, firingRate, trainingData(i,k).firingRates);
    end

    v_predict(k).reachingAngle = lsqminnorm(firingRate',vel');
end

spikes = [];
reachingAngle = [];
spikeCount = zeros(length(training_data),98);

for k = 1:8
    for i = 1:98
        for n = 1:length(training_data)
            number_of_spikes = length(find(training_data(n,k).spikes(i,1:320)==1));
            spikeCount(n,i) = number_of_spikes;
        end
    end
    spikes = cat(1, spikes, spikeCount);
    reaching_angle(1:length(training_data)) = k;
    reachingAngle = cat(2, reachingAngle, reaching_angle);
end

% Neural network to predict angle (classification)
layer_sizes = [98, 70, 8];
maxIter = 300; %400
lambda = 0.6; %0.8 %regularisation
alpha = 0.4; %0.3 %learning rate
% loss = [];
% count = [];
% precisionT = [];
% m = size(spikes,1);


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

%         if mod(i,20) == 0
%             loss = [loss;J];
%             count = [count;i];
%             plot(ax1,count, loss, 'LineWidth', 2); ax1.YGrid = "on"; ax1.XGrid = "on";
%             title(ax1,'Loss');
%             xlabel(ax1,'Iterations');
%             ylabel(ax1,'Loss function')
%             pred = nnpredict(weights, spikes, layer_sizes);
%             precision = mean(double(pred == reachingAngle')) * 100;
%             precisionT = [precisionT;precision];
%             plot(ax2,count, precisionT, 'LineWidth', 2); ax2.YGrid = "on"; ax2.XGrid = "on";
%             title(ax2,'Accuracy');
%             xlabel(ax2,'Iterations');
%             ylabel(ax2,'Accuracy');
%             disp(['Iteration #: ' num2str(i) ' / ' num2str(maxIter) ' | Loss J: ' num2str(J) ' | Accuracy: ' ...
%                     num2str(precision)]);
%             drawnow();
%         end
end

finT = toc;

disp(['Time spent on training the net: ' num2str(finT) ' seconds' ])

modelParameters = struct('v_predict',v_predict,'nnweights',weights);

end

% Functions for the neural network.

function W = initweights(L_in, L_out)
%W = zeros(L_out, 1 + L_in);
epsilon_init = 0.12; %0.12
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