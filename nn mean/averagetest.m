clc
close all
clear all

load monkeydata_training.mat

averages = zeros(2,8,1000);
counts = zeros(2,8,1000);

for k=1:8
    for n=1:100
        for idx=1:length(trial(n,k).handPos)
            averages(1,k,idx) = averages(1,k,idx) + trial(n,k).handPos(1,idx);
            averages(2,k,idx) = averages(2,k,idx) + trial(n,k).handPos(2,idx);
        end
        counts(1,k,1:length(trial(n,k).handPos)) = counts(1,k,1:length(trial(n,k).handPos)) + 1;
        counts(2,k,1:length(trial(n,k).handPos)) = counts(2,k,1:length(trial(n,k).handPos)) + 1;
    end
end

averages = averages ./ 100;