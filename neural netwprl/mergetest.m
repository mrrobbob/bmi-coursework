%load trial data
load("monkeydata_training.mat");

rng(2023);
ix = randperm(length(trial));

% Select training and testing data (you can choose to split your data in a different way if you wish)
trial_train = trial(ix(1:50),:);
trial_test = trial(ix(51:end),:);

%compute firing rate of each neuron in each trial
% not sure if firing rates should be normalized
firing_rates= cell(size(trial));
for i= 1:height(trial)
    for j= 1:8
        firing_rates{i,j}= mean(trial(i,j).spikes(:,:),2)';
    end
end

%partition trial data 
train_ratio= 0.6;
num_folds= 500;
% test_ratio= 0.2;

cvp = cvpartition(height(trial), 'KFold', num_folds)

layer_sizes = [98,150,8];

% initialize the weights randomly [the weights have the bias inside on the first column]
for i = 1:(size(layer_sizes, 2)-1)
    initial_weights{i} = randInitializeWeights(layer_sizes(i), layer_sizes(i+1));
end

% Weight regularization parameter and learning rate
lambda = 1;
alfa = 0.1;

%maxIter = 1000;

loss = [];
count = [];
precisionT = [];
weights = initial_weights;

figure;
ax1 = subplot(2,1,1); ax1.YGrid = "on"; ax1.XGrid = "on";
ax2 = subplot(2,1,2); ax2.YGrid = "on"; ax2.XGrid = "on";

tic; 
for i = 1:500

%     train_idx= cvp.training(i);
%     test_idx= cvp.test(i);

%     trial_train= firing_rates(train_idx,:);
%     trial_test= firing_rates(test_idx,:);

    trial_train=trial(ix(1:50),:);
    trial_test=trial(ix(51:end),:);

    %create variable Ytrain and Xtrain
    Xtrain=zeros(numel(trial_train),98);
    Ytrain=zeros(numel(trial_train),1);
    for j= 1:numel(trial_train)
        Xtrain(j,:)=trial_train{j};
        Ytrain(j)= ceil(j/length(trial_train));
    end
    %create variable Ytest and Xtest
    Xtest=zeros(numel(trial_test),98);
    Ytest=zeros(numel(trial_test),1);
    for j= 1:numel(trial_test)
        Xtest(j,:)=trial_test{j};
        Ytest(j)= ceil(j/length(trial_test));
    end
    [J, weights] = train(weights, layer_sizes,Xtrain, Ytrain, lambda, alfa);
                                   
    if mod(i,25) == 10
        loss = [loss;J];
        count = [count;i];
        plot(ax1,count, loss, 'LineWidth', 2); ax1.YGrid = "on"; ax1.XGrid = "on";
        title(ax1,'Loss EVOLUTION');
        xlabel(ax1,'Iterations');
        ylabel(ax1,'Loss function')
        pred = predict(weights, Xtest, layer_sizes);
        precision = mean(double(pred == Ytest)) * 100;
        precisionT = [precisionT;precision];
        plot(ax2,count, precisionT, 'LineWidth', 2); ax2.YGrid = "on"; ax2.XGrid = "on";
        title(ax2,'Accuracy EVOLUTION');
        xlabel(ax2,'Iterations');
        ylabel(ax2,'Accuracy');
        disp(['Iteration #: ' num2str(i) ' / ' num2str(num_folds) ' | Loss J: ' num2str(J) ' | Accuracy: ' ...
                num2str(precision)]);
        drawnow();
    end
    
end

finT = toc;


disp(['Time spent on training the net: ' num2str(finT) ' seconds' ])

figure;
i = randi(length(Ytest));
pred = predict(weights, Xtest(i,:), layer_sizes);
imshow(reshape(Xtest(i,:),20,20));
fprintf('True class: %d  |  Predicted class: %d\n',Ytest(i),pred);