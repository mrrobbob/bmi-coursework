function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
% Find number of spikes
fs = 1000;
spikeCount = zeros(98,1);
if length(test_data.spikes) <= 320
    for i = 1:98
        numspikes = length(find(test_data.spikes(i,1:320)==1));
        spikeCount(i) = numspikes;
    end
    % Find direction with most predictions (majority vote) and set as reaching angle
    direction = mode(nnpredict({modelParameters(1).nnweights,modelParameters(2).nnweights},spikeCount',[98,70,8]));
else
    % Use previous value
    direction = modelParameters.direction;
end



% % Predict movement.
% timegap = 20;
% % 20 ms gaps.
% t_0 = length(test_data.spikes)-timegap;
% t_1 = length(test_data.spikes);
% 
% % Calculate firing rate
% firingRate = zeros(98,1);
% for i = 1:98
%     numspikes = length(find(test_data.spikes(i,t_0:t_1)==1));
%     firingRate(i) = numspikes * fs / timegap;
% end
% 
% % Estimate velocity.
% velocity_x = firingRate'*modelParameters(1).v_predict(direction).reachingAngle(:,1);
% velocity_y = firingRate'*modelParameters(1).v_predict(direction).reachingAngle(:,2);
% 
% % Output
% 
% if length(test_data.spikes) <= 320
%     x = test_data.startHandPos(1);
%     y = test_data.startHandPos(2);
% else
%     % s = s_0 + v*t
%     x = test_data.decodedHandPos(1,length(test_data.decodedHandPos(1,:))) + velocity_x / fs * timegap;
%     y = test_data.decodedHandPos(2,length(test_data.decodedHandPos(2,:))) + velocity_y / fs * timegap;
% end

newModelParameters(1).v_predict = modelParameters(1).v_predict;
newModelParameters(1).nnweights = modelParameters(1).nnweights;
newModelParameters(2).nnweights = modelParameters(2).nnweights;
newModelParameters(1).direction = direction;

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