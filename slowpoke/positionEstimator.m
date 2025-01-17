function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)

spikeCount = zeros(98,1);

if length(test_data.spikes) <= 320
    % Find number of spikes
    for i = 1:98
        numspikes = length(find(test_data.spikes(i,1:320)==1));
        spikeCount(i) = numspikes;
    end

    % Find direction with most predictions (majority vote) and set as reaching angle
    direction = mode(nnpredict(modelParameters.nnweights,spikeCount));
    
else
    % use value that was found prevously
    direction = modelParameters.direction;
end

%% predict movement

% current time window
tmin = length(test_data.spikes)-20;
tmax = length(test_data.spikes);

% Calculate firing rate
firingRate = zeros(98,1);
for i = 1:98
    numspikes = length(find(test_data.spikes(i,tmin:tmax)==1));
    firingRate(i) = numspikes/(20*0.001);
end

% Estimate velocity
velocity_x = firingRate'*modelParameters(1).beta(direction).reachingAngle(:,1);
velocity_y = firingRate'*modelParameters(1).beta(direction).reachingAngle(:,2);

%% output

if length(test_data.spikes) <= 320
    x = test_data.startHandPos(1);
    y = test_data.startHandPos(2);
else
    % s = s_0 + v * t
    x = test_data.decodedHandPos(1,length(test_data.decodedHandPos(1,:))) + velocity_x*(20*0.001);
    y = test_data.decodedHandPos(2,length(test_data.decodedHandPos(2,:))) + velocity_y*(20*0.001);
end

newModelParameters(1).beta = modelParameters(1).beta;
newModelParameters(1).nnweights = modelParameters(1).nnweights;
newModelParameters(2).nnweights = modelParameters(2).nnweights;
newModelParameters(1).direction = direction;

end