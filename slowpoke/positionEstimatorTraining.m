% load("monkeydata0.mat");
% rng(2023);
% ix = randperm(length(trial));
% training_data = trial(ix(1:50),:);
function [modelParameters] = positionEstimatorTraining(training_data)

%% Find firing rates

spike_rate = [];
firingRates = [];
xVelArray = [];
yVelArray = [];

trainingData = struct([]);
velocity = struct([]);

dt = 10; % bin size

for k = 1:8
    for i = 1:98
        for n = 1:length(training_data)
            for t = 300:dt:550-dt
                
                % find the firing rates of one neural unit for one trial
                number_of_spikes = length(find(training_data(n,k).spikes(i,t:t+dt)==1));
                spike_rate = cat(2, spike_rate, number_of_spikes/(dt*0.001));
                
                % find the velocity of the hand movement
                % (needs calculating just once for each trial)
                if i==1
                    x_low = training_data(n,k).handPos(1,t);
                    x_high = training_data(n,k).handPos(1,t+dt);
                    
                    y_low = training_data(n,k).handPos(2,t);
                    y_high = training_data(n,k).handPos(2,t+dt);
                    
                    x_vel = (x_high - x_low) / (dt*0.001);
                    y_vel = (y_high - y_low) / (dt*0.001);
                    xVelArray = cat(2, xVelArray, x_vel);
                    yVelArray = cat(2, yVelArray, y_vel);
                end
                
            end
            
            % store firing rate of one neural unit for every trial in one array
            firingRates = cat(2, firingRates, spike_rate);
            spike_rate = [];
            
        end
        
        trainingData(i,k).firingRates = firingRates;
        velocity(k).x = xVelArray;
        velocity(k).y = yVelArray;
        
        firingRates = [];
        
    end
    xVelArray = [];
    yVelArray = [];
end

%% Linear Regression
% used to predict velocity
beta = struct([]);

for k=1:8
    
    vel = [velocity(k).x; velocity(k).y];
    firingRate = [];
    for i=1:98
    firingRate = cat(1, firingRate, trainingData(i,k).firingRates);
    end
    
    beta(k).reachingAngle = lsqminnorm(firingRate',vel');
    
end

%% KNN Classifier
% used to predict the reaching angle from the first 320ms

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

layer_sizes = [98, 73, 8];
maxIter = 1000;
lambda = 0.8;
alfa = 0.05;
loss = [];
count = [];
precisionT = [];
m = size(spikes,1);


for i = 1:(size(layer_sizes, 2)-1)
    initial_weights{i} = randInitializeWeights(layer_sizes(i), layer_sizes(i+1));
end

weights = initial_weights;

figure(1);
ax1 = subplot(2,1,1); ax1.YGrid = "on"; ax1.XGrid = "on";
ax2 = subplot(2,1,2); ax2.YGrid = "on"; ax2.XGrid = "on";

reachingAnglenew = (1:8) == reachingAngle';

tic; 
for i = 1:maxIter
    [J, weights] = train(weights, layer_sizes,spikes, reachingAnglenew, lambda, alfa);
                                   
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



%knn = fitcknn(spikes,reachingAngle);

%modelParameters = struct('beta',beta,'knnModel',knn); 

modelParameters = struct('beta',beta,'nnweights',weights);

end

% rr = reachingAngle';
% i = randi(length(reachingAngle'));
% pred = nnpredict(weights, spikes(i,:), layer_sizes);
% fprintf('True class: %d  |  Predicted class: %d\n',rr(i),pred);