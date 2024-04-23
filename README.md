# BMI Coursework

## Overview

This repository contains all the code written for the Brain Machine Interfaces university module coursework for my 4th year undergraduate Biomedical Engineering degree at Imperial College London.

## Task

As a group, we were tasked with designing a neural decoder to drive a hypothetical prosthetic device. We were given spike train data recorded from a monkey's brain, as it repeatedly performs a centre-out reach task. We had to design an algorithm to estimate the precise trajectory of the monkey's hand as it reached the target. 
The x and y positions of the monkey's hand must be estimated over time. z-axis data was provided but disregarded.

### Constraints
- No "time travel". The estimator must be causal i.e. it has no access to data in the future.
- Use of any MATLAB resources outside vanilla MATLAB was forbidden. There was no access to external machine learning libraries and all code must be written from scratch.

The error and computation time of all estimators in the cohort were placed on a leaderboard as a competition. Total error was weighted 80% by position error and 20% by computation time. I wrote the first iteration of the code and it remained (nearly; with tweaked hyperparameters) in its entirety throughout the competition. It was placed 6th overall.

## Solution

I decided to split the task into two: one part for reaching angle classification and another for position trajectory estimation. A neural network was written from scratch to predict the reach angle and linear regression was used to estimate kinematic position.

## Evaluation

The final code submitted was not optimised. Using a neural network for simple angle classification is overkill and other faster techniques (such as K-Nearest Neighbours (KNN) in combination with Principal Components Analysis (PCA)) would work better. Conversely, using linear regression for kinematic estimation was short-sighted. Regression linked spike frequency with velocity, and the neural data contained much more information than what was used. Overall, the code was a good first try but much more could have been tried and implemented.
