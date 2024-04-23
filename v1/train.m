function [J, newWeights] = train(weights, layer_sizes,X, y_new, lambda, alfa)
    % Setup some useful variables
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
        
        newWeights{1,i}(:,1) = weights{1,i}(:,1) - alfa*mean(weight_grad{1,i}(:,1),1);
        newWeights{1,i}(:,2:end) = weights{1,i}(:,2:end) - alfa*weight_grad{1,i}(:,2:end);
        
    end
end