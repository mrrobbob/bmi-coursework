function [model] = estimationtrainer(training_data)
num_angles = 8;
num_neurons = 98;

%rate_data = struct([]);
firing_rate = [];
spike_rate = [];
velocities = struct("x",[],"y",[]);
v_x = [];
v_y = [];

bin_size = 5;
for k=1:num_angles
    for i=1:num_neurons
        for n=1:length(training_data)
            for j=300:bin_size:550-bin_size
                num_spikes = length(find(training_data(n,k).spikes(i,j:j+bin_size)==1));
                spike_rate = cat(2,spike_rate,num_spikes/(bin_size*0.001));

                if i==1

                    v_x_ = (training_data(n,k).handPos(1,j+bin_size) - training_data(n,k).handPos(1,j)) / bin_size / 0.001;
                    v_y_ = (training_data(n,k).handPos(2,j+bin_size) - training_data(n,k).handPos(2,j)) / bin_size / 0.001;
                    v_x = cat(2,v_x,v_x_);
                    v_y = cat(2,v_y,v_y_);
                end
            end
            firing_rate = cat(2,firing_rate,spike_rate);
            spike_rate = [];
        end
        training_data(i,k).firingrate = firing_rate;
        velocities(k).x = v_x;
        velocities(k).y = v_y;
        firing_rate = [];
    end
    v_x = [];
    v_y = [];

end

predict_v = struct([]);

for k=1:num_angles
    rate_fire = [];
    v = [velocities(k).x;velocities(k).y];
    for i=1:num_neurons
        rate_fire = cat(1,rate_fire,training_data(i,k).firing_rate);
    end
    predict_v(k).angle = lsqminnorm(rate_fire',v');
end

spikes = [];
angle = [];
num_spikes = zeros(length(training_data),num_neurons);

for k=1:num_angles
    for i=1:num_neurons
        for n=1:length(training_data)
            num_spikes(n,i) = length(find(training_data(n,k).spikes(i,1:320)==1));
        end
    end
    spikes = cat(1,spikes,num_spikes);
    angle_(1:length(training_data)) = k;
    angle = cat(2,angle,angle_);
end

layer_sizes = [98, 73, 8];
maxIter = 1000;
lambda = 0.8;
alpha = 0.05;
loss = [];
count = [];
precisionT = [];

for i = 1:(size(layer_sizes, 2)-1)
    initial_weights{i} = randInitializeWeights(layer_sizes(i), layer_sizes(i+1));
end

weights = initial_weights;

figure(1);
ax1 = subplot(2,1,1); ax1.YGrid = "on"; ax1.XGrid = "on";
ax2 = subplot(2,1,2); ax2.YGrid = "on"; ax2.XGrid = "on";

reachingAnglenew = (1:8) == reachingAngle';

% tic; 
for i = 1:maxIter
    [J, weights] = train(weights, layer_sizes,spikes, reachingAnglenew, lambda, alpha);
                                   
    if mod(i,20) == 0
        loss = [loss;J];
        count = [count;i];
        plot(ax1,count, loss, 'LineWidth', 2); ax1.YGrid = "on"; ax1.XGrid = "on";
        title(ax1,'Loss EVOLUTION');
        xlabel(ax1,'Iterations');
        ylabel(ax1,'Loss function')
        pred = nnpredict(weights, spikes, layer_sizes);
        precision = mean(double(pred == reachingAngle')) * 100;
        precisionT = [precisionT;precision];
        plot(ax2,count, precisionT, 'LineWidth', 2); ax2.YGrid = "on"; ax2.XGrid = "on";
        title(ax2,'Accuracy EVOLUTION');
        xlabel(ax2,'Iterations');
        ylabel(ax2,'Accuracy');
        disp(['Iteration #: ' num2str(i) ' / ' num2str(maxIter) ' | Loss J: ' num2str(J) ' | Accuracy: ' ...
                num2str(precision)]);
        drawnow();
    end
    
end

finT = toc;
disp(['Time spent on training the net: ' num2str(finT) ' seconds' ])

model = struct('beta',beta,'nnweights',weights);

end

function [J, newWeights] = train(weights, layer_sizes,X, y_new, lambda, alfa)
    m = size(X, 1);
    a1 = [ones(m, 1) X];   
    J = 0;
    % Initialise sum and activation.
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

    % Cost function for classification
    J = sum(sum(-y_new .* log(a{1,end}) - (1 - y_new) .* log(1 - a{1,end}))) / m ;

    % add regularization to the cost function
    regularisation = 0;
    for i = 1:(size(layer_sizes,2)-1)
        regularisation = regularisation + sum(weights{1,i}(:,2:end).^2,'all');
    end

    regularisation =  lambda/(2*m) * regularisation;
    J = J + regularisation;
    

    % Back propagation 

    % Sigmas
    for i = size(layer_sizes,2):-1:2
        if i == size(layer_sizes,2)
            s{i} = a{1,i} - y_new;
        else
            s{i} = (s{1,i+1}*weights{1,i}(:, 2:end)) .* sigmoidGradient(z{1,i});
        end
        
    end
    % Deltas
    for i = 1:(size(layer_sizes,2)-1)
        deltas{i} = s{1,i+1}'*a{1,i};
    end

    % Gradient descent
    for i = 1:(size(layer_sizes,2)-1) 
        p = lambda * [zeros(size(weights{1,i}, 1), 1), weights{1,i}(:, 2:end)] / m;
        weight_grad{i} = deltas{1,i}./m + p;

        newWeights{i} = zeros(size(weights{1,i}));
        
        newWeights{1,i}(:,1) = weights{1,i}(:,1) - alfa*mean(weight_grad{1,i}(:,1),1);
        newWeights{1,i}(:,2:end) = weights{1,i}(:,2:end) - alfa*weight_grad{1,i}(:,2:end);
    end
end

function p = predict(weights, X, layer_sizes)
m = size(X, 1);
p = zeros(size(X, 1), 1);

% Initialize sum and activation.
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

function W = randInitializeWeights(L_in, L_out)
W = zeros(L_out, 1 + L_in);
epsilon_init = 0.10;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
end

function sig = sigmoid(z)
    sig = 1.0 ./ (1.0 + exp(-z));
end

function sigGrad = sigmoidGradient(z)
sigGrad = zeros(size(z));
s = sigmoid(z);
sigGrad = s.*(1-s);
end

