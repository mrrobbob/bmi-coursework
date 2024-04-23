%generate monte carlo simulations
N= 10; % number of b.neurons per ensemble. Do 10 first then increase. Watch running time
rng('default');

%load trial data
load("monkeydata_training.mat");

%compute firing rate of each neuron in each trial
% not sure if firing rates should be normalized
firing_rates= cell(size(trial));
for i= 1:height(trial)
    for j= 1:8
        firing_rates{i,j}= mean(trial(i,j).spikes(:,:),2)';
    end
end

% firing_rates= cell2mat(firing_rates);

% % set seed
% rng(43);

%partition trial data 
train_ratio= 0.6;
num_folds= 3;
% test_ratio= 0.2;

cvp = cvpartition(height(trial), 'KFold', num_folds)


for i= 1:num_folds
    train_idx= cvp.training(i);
    test_idx= cvp.test(i);

    trial_train= firing_rates(train_idx,:);
    trial_test= firing_rates(test_idx,:);
    
%     % get the smallest length of spikes data in the training set 
%      min_length=Inf;
%      for k= 1:numel(trial_train)
%              min_length= min(min_length, length(trial_train(k).spikes));
%      end

%     % create variable Y (movement direction)
%     Y= zeros(numel(trial_train),1);
%     for z= 0:7
%         start_index= 1+height(trial_train)*z;
%         end_index= height(trial_train)*(z+1);
%         Y(start_index:end_index)= ones(height(trial_train),1).*(z+1);
%     end
%     % concatenate neuronal firing rates (variable X) from trial_train
%     Xtrain= [];
%     for j= 1:numel(trial_train)
%         Xtrain= [Xtrain; trial_train{j}];
%     end

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

%     train network using an ANN (single layer feedforward net)
    architecture = [93,150,1];
    

    net= fitcnet(Xtrain, Ytrain, 'LayerSizes', 150);
    loss_val=loss(net, Xtest, Ytest, "LossFun", "classiferror");
    accuracy= 1-loss_val

end

% % generate 10 possible combinations
% N=10;
% 
% % generate random combinations of number 1-98 of random length
% neurons_length=randi(98,1);
% rand_neurons= sort(randi(98,1, neurons_length));
% 
% for i= 1:num_folds
%     train_idx= cvp.training(i);
%     test_idx= cvp.test(i);
%     
%     trial_train= firing_rates(train_idx,:);
%     trial_test= firing_rates(test_idx,:);
%     
% 
%     %create variable Ytrain and Xtrain
%     Xtrain=zeros(numel(trial_train),neurons_length);
%     Ytrain=zeros(numel(trial_train),1);
%     for j= 1:numel(trial_train)
%         Xtrain(j,1:numel(rand_neurons))=trial_train{j}(rand_neurons);
%         Ytrain(j)= ceil(j/length(trial_train));
%     end
%     %create variable Ytest and Xtest
%     Xtest=zeros(numel(trial_test),neurons_length);
%     Ytest=zeros(numel(trial_test),1);
%     for j= 1:numel(trial_test)
%         Xtest(j,1:numel(rand_neurons))=trial_test{j}(rand_neurons);
%         Ytest(j)= ceil(j/length(trial_test));
%     end
% 
% %     train network using an ANN (single layer feedforward net)
% %     net= fitcnet(Xtrain, Ytrain, 'LayerSizes', 150, 'Activations','sigmoid');
% %     net= configure(net, Xtrain, Ytrain);
% %     net.layers{end}.transferFcn='linear';
% %     net.outputFcn='purelin';
%     net= patternnet(150, 'trainscg','mse');
%     net.layers{1}.transferFcn= 'logsig';
%     net.layers{2}.transferFcn= 'purelin';
% %     view(net)
%     [net,~,~,dY]= train(net, Xtrain', Ytrain');
%     net.LW{end}
%     
% %     loss_val=loss(net, Xtest, Ytest, "LossFun", "classiferror");
% %     accuracy= 1-loss_val
%     ypred= net(Xtest');
%     perf= perform(net, ypred', Ytest');
% 
%     %local sensitivity analysis
%     %jacob= dY*net.LW{2}
% end