%% FULL LSTM REGRESSION FOR NASA LFP SOC
clear; clc; close all;

% --- 1. LOAD PREPROCESSED DATA ---
if ~exist('NASA_LFP_Preprocessed.mat', 'file')
    error('Data file not found! Please run the Deep-Dive Preprocessing script first.');
end
fprintf('Loading 333,773 sequences...\n');
load('NASA_LFP_Preprocessed.mat'); 

% --- 2. DATA SPLITTING (80/10/10) - Same seed for comparison ---
rng(42); % Fixed seed ensures XTest is the same as your CNN test set
n = length(YData);
idx = randperm(n);
idxTrain = idx(1 : round(0.8*n));
idxVal   = idx(round(0.8*n)+1 : round(0.9*n));
idxTest  = idx(round(0.9*n)+1 : end);

XTrain = XData(idxTrain); YTrain = YData(idxTrain);
XVal   = XData(idxVal);   YVal   = YData(idxVal);
XTest  = XData(idxTest);  YTest  = YData(idxTest);

% --- 3. LSTM ARCHITECTURE ---
% LSTMs process the sequence to find temporal dependencies in LFP voltage

layers = [
    sequenceInputLayer(3, 'Name', 'input')
    
    lstmLayer(128, 'OutputMode', 'last', 'Name', 'lstm1')
    batchNormalizationLayer('Name', 'bn1')
    dropoutLayer(0.2, 'Name', 'drop1')
    
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    
    fullyConnectedLayer(1, 'Name', 'fc_out')
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
fprintf('Training Pure LSTM Model on LFP Data...\n');
net_lstm = trainNetwork(XTrain, YTrain, layers, options);

% --- 6. EVALUATION ---
fprintf('Post-processing results...\n');
YPred = predict(net_lstm, XTest);
YTrue = YTest;
errors = YTrue - YPred;
rmseVal = sqrt(mean(errors.^2));
maeVal = mean(abs(errors));
accuracy = (1 - maeVal) * 100;


% --- 7. MULTI-TAB VISUALIZATION (FIXED FOR FAIR COMPARISON) ---
hFig = figure('Name', 'NASA LFP LSTM Results', 'Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
hTabGroup = uitabgroup(hFig);

% Tab 1: Prediction Comparison (Zoomed in to match NMC style)
tab1 = uitab(hTabGroup, 'Title', 'Predictions');
axes1 = axes('Parent', tab1);

% Define a subset to match the NMC plot density (e.g., first 300 samples)
zoomRange = 1:300; 

subplot(2,1,1, axes1);
plot(YTrue(zoomRange), 'b-', 'LineWidth', 1.5); hold on;
plot(YPred(zoomRange), 'r--', 'LineWidth', 1.2);
ylabel('State of Charge (0-1)');
title(['LSTM SoC Estimation (Subset of ', num2str(length(zoomRange)), ' Samples)']);
legend('True SoC', 'Predicted SoC');
grid on;
xlim([0 length(zoomRange)]); % This makes it look exactly like your NMC plot

% Subplot 2: Full Test Set (To show the overall accuracy)
subplot(2,1,2, axes1);
plot(YTrue, 'b', 'LineWidth', 0.5); hold on;
plot(YPred, 'r--', 'LineWidth', 0.5);
ylabel('SOC (0-1)'); xlabel('Total Test Samples');
title(['Final LSTM SOC Accuracy: ', num2str(accuracy, '%.2f'), '% (RMSE: ', num2str(rmseVal, '%.4f'), ')']);
grid on;

% ... (Keep Tab 2: Error Analysis as it was) ...


% Tab 2: Scatter Plot & Error Histogram
tab2 = uitab(hTabGroup, 'Title', 'Error Analysis');
axes2a = subplot(1,2,1, 'Parent', tab2);
scatter(axes2a, YTrue, YPred, 5, 'filled', 'MarkerFaceAlpha', 0.1);
hold(axes2a, 'on');
plot(axes2a, [0 1], [0 1], 'r', 'LineWidth', 2);
xlabel(axes2a, 'True SOC'); ylabel(axes2a, 'Predicted SOC');
title(axes2a, 'Regression Scatter Plot');
grid(axes2a, 'on');

axes2b = subplot(1,2,2, 'Parent', tab2);
histogram(axes2b, errors, 50, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.8]);
xlabel(axes2b, 'Error (True - Predicted)'); ylabel(axes2b, 'Probability Density');
title(axes2b, 'Error Histogram');
grid(axes2b, 'on');

fprintf('\n--- Final LSTM Results ---\n');
fprintf('RMSE: %.4f\n', rmseVal);
fprintf('Accuracy: %.2f%%\n', accuracy);

save('Trained_LSTM_LFP.mat', 'net_lstm');