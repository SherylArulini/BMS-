%% FULL PURE CNN REGRESSION FOR NASA LFP SOC
clear; clc; close all;

% --- 1. LOAD PREPROCESSED DATA ---
if ~exist('NASA_LFP_Preprocessed.mat', 'file')
    error('Data file not found! Please run the Deep-Dive Preprocessing script first.');
end
fprintf('Loading 333,773 sequences...\n');
load('NASA_LFP_Preprocessed.mat'); 

% --- 2. DATA SPLITTING (80/10/10) ---
n = length(YData);
idx = randperm(n);
idxTrain = idx(1 : round(0.8*n));
idxVal   = idx(round(0.8*n)+1 : round(0.9*n));
idxTest  = idx(round(0.9*n)+1 : end);

XTrain = XData(idxTrain); YTrain = YData(idxTrain);
XVal   = XData(idxVal);   YVal   = YData(idxVal);
XTest  = XData(idxTest);  YTest  = YData(idxTest);
windowSize = size(XData{1}, 2); % Should be 50

% --- 3. CNN ARCHITECTURE (FIXED) ---

layers = [
    sequenceInputLayer(3, 'MinLength', windowSize, 'Name', 'input') 
    
    convolution1dLayer(7, 64, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool1') 
    
    convolution1dLayer(5, 128, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool2') 
    
    convolution1dLayer(3, 256, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    
    globalAveragePooling1dLayer('Name', 'gap') 
    
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.2, 'Name', 'drop')
    
    fullyConnectedLayer(1, 'Name', 'fc2')
    regressionLayer('Name', 'out')];

% --- 4. TRAINING OPTIONS ---
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 1024, ... 
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 20, ...
    'ValidationData', {XVal, YVal}, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false);

% --- 5. TRAINING ---
fprintf('Training Pure CNN Model on LFP Data...\n');
net_cnn = trainNetwork(XTrain, YTrain, layers, options);

% --- 6. EVALUATION ---
fprintf('Post-processing results...\n');
YPred = predict(net_cnn, XTest);
YTrue = YTest;

rmseVal = sqrt(mean((YTrue - YPred).^2));
maeVal = mean(abs(YTrue - YPred));
accuracy = (1 - maeVal) * 100;

% --- 7. PLOTTING (MODIFIED FOR FAIR COMPARISON) ---
figure('Name', 'CNN LFP Prediction Results', 'Color', 'w', 'Position', [100 100 800 600]);

% Select a subset for the top plot to avoid clutter
zoomRange = 1:min(300, length(YTrue)); 

% Top Plot: Point-by-point tracking
subplot(2,1,1);
plot(YTrue(zoomRange), 'b-o', 'MarkerSize', 4, 'LineWidth', 1); hold on;
plot(YPred(zoomRange), 'r--x', 'MarkerSize', 4, 'LineWidth', 1);
ylabel('State of Charge (0-1)');
title(['LFP SOC Estimation (Subset of ', num2str(length(zoomRange)), ' Samples)']);
legend('True Value', 'CNN Prediction');
grid on;

% Bottom Plot: Trend tracking
subplot(2,1,2);
plot(YTrue(zoomRange), 'b', 'LineWidth', 1.5); hold on;
plot(YPred(zoomRange), 'r--', 'LineWidth', 1.5);
ylabel('SoC (0-1)');
xlabel('Test Samples');
title(['LFP SoC Accuracy: ', num2str(accuracy, '%.2f'), '% (RMSE: ', num2str(rmseVal, '%.4f'), ')']);
legend('True SoC', 'Predicted SoC');
grid on;

fprintf('\n--- Final Results ---\n');
fprintf('RMSE: %.4f\n', rmseVal);
fprintf('Accuracy: %.2f%%\n', accuracy);

save('Trained_CNN_LFP.mat', 'net_cnn');