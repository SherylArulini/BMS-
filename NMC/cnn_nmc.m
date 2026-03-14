%% CNN REGRESSION FOR NMC SOC AND CAPACITY ESTIMATION
clear; clc; close all;

% --- 1. CONFIGURATION & LOADING ---
% Update these paths to match your local machine
basePath = 'C:\Matlab Projects\bms\Raw NMC'; 
dataFolder = fullfile(basePath, 'data');
metadata = readtable(fullfile(basePath, 'metadata.csv'));

fixedTimeSteps = 200; 
featureNames = {'Voltage_measured', 'Current_measured', 'Temperature_measured'};

% Filter for discharge cycles
dischargeMeta = metadata(strcmp(metadata.type, 'discharge'), :);
numSamples = height(dischargeMeta);
X_raw = zeros(numSamples, fixedTimeSteps, 3);
Y_raw = zeros(numSamples, 1);

fprintf('Preprocessing Data (Moving Average + Interpolation)...\n');
for i = 1:numSamples
    filePath = fullfile(dataFolder, dischargeMeta.filename{i});
    if exist(filePath, 'file')
        data = readtable(filePath);
        rawS = data{:, featureNames};
        
        % Noise reduction
        rawS = movmean(rawS, 5); 
        
        % Resampling to fixed length (200 steps)
        oldSteps = size(rawS, 1);
        newSteps = linspace(1, oldSteps, fixedTimeSteps);
        for f = 1:3
            X_raw(i, :, f) = interp1(1:oldSteps, rawS(:, f), newSteps, 'linear');
        end
        Y_raw(i) = dischargeMeta.Capacity(i);
    end
end

% --- 2. CLEANING & NORMALIZATION ---
nanIdx = isnan(Y_raw) | any(any(isnan(X_raw), 2), 3);
X_raw(nanIdx, :, :) = []; 
Y_raw(nanIdx) = [];

% Feature Normalization (0 to 1)
X_norm = X_raw;
for f = 1:3
    fMin = min(X_norm(:,:,f), [], 'all'); 
    fMax = max(X_norm(:,:,f), [], 'all');
    X_norm(:,:,f) = (X_norm(:,:,f) - fMin) / (fMax - fMin);
end

% Target Normalization
capMin = min(Y_raw); 
capMax = max(Y_raw);
Y_scaled = (Y_raw - capMin) / (capMax - capMin);

% --- 3. DATA SPLITTING (80/10/10) ---
n = length(Y_raw);
idx = randperm(n);
idxTrain = idx(1 : round(0.8*n));
idxVal   = idx(round(0.8*n)+1 : round(0.9*n));
idxTest  = idx(round(0.9*n)+1 : end);

% Convert to Cell Array format for sequenceInputLayer
toCell = @(data) arrayfun(@(i) squeeze(data(i,:,:))', 1:size(data,1), 'UniformOutput', false)';
XTrain = toCell(X_norm(idxTrain, :, :)); YTrain = Y_scaled(idxTrain);
XVal   = toCell(X_norm(idxVal, :, :));   YVal   = Y_scaled(idxVal);
XTest  = toCell(X_norm(idxTest, :, :));  YTest  = Y_scaled(idxTest);

% --- 4. CNN ARCHITECTURE ---
% Optimized for 1D Battery Time-Series
layers = [
    sequenceInputLayer(3, 'MinLength', fixedTimeSteps, 'Name', 'input')
    
    convolution1dLayer(7, 64, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool1') % 200 -> 100
    
    convolution1dLayer(5, 128, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool2') % 100 -> 50
    
    convolution1dLayer(3, 256, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    
    globalAveragePooling1dLayer('Name', 'gap')
    
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.2, 'Name', 'drop')
    fullyConnectedLayer(1, 'Name', 'fc2')
    regressionLayer('Name', 'out')];

% --- 5. TRAINING OPTIONS ---
options = trainingOptions('adam', ...
    'MaxEpochs', 250, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 100, ...
    'L2Regularization', 0.0005, ...
    'ValidationData', {XVal, YVal}, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false);

fprintf('Training Pure CNN Model...\n');
net = trainNetwork(XTrain, YTrain, layers, options);

% --- 6. POST-PROCESSING & METRICS ---
YPred_norm = predict(net, XTest);
YPred_Ah = YPred_norm * (capMax - capMin) + capMin;
YTrue_Ah = Y_raw(idxTest);

% SoC Calculation based on Rated Capacity
ratedCap = max(Y_raw); 
SoC_True = YTrue_Ah / ratedCap;
SoC_Pred = YPred_Ah / ratedCap;

% Accuracy metric
validIdx = SoC_True > 0.05;
accuracy = (1 - mean(abs(SoC_True(validIdx) - SoC_Pred(validIdx)) ./ SoC_True(validIdx))) * 100;

% --- 7. PLOTTING ---
figure('Name', 'CNN Prediction Results', 'Color', 'w', 'Position', [100 100 800 600]);

subplot(2,1,1);
plot(YTrue_Ah, 'b-o', 'MarkerSize', 4, 'LineWidth', 1); hold on;
plot(YPred_Ah, 'r--x', 'MarkerSize', 4, 'LineWidth', 1);
ylabel('Capacity (Ah)');
title('NMC Capacity Estimation');
legend('True Value', 'CNN Prediction');
grid on;

subplot(2,1,2);
plot(SoC_True, 'b', 'LineWidth', 1.5); hold on;
plot(SoC_Pred, 'r--', 'LineWidth', 1.5);
ylabel('SoC (0-1)');
xlabel('Test Samples');
title(['SoC Accuracy: ', num2str(accuracy, '%.2f'), '%']);
legend('True SoC', 'CNN Predicted SoC');
grid on;

fprintf('Final Test Accuracy: %.2f%%\n', accuracy);