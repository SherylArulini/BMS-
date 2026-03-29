
clear; clc; close all;

% --- 1. LOAD DATA ---
load('NASA_LFP_Preprocessed.mat'); 

% --- 2. ENHANCED FEATURE ENGINEERING ---
% Adding Cumulative Current (Feature 4) to bypass the LFP voltage plateau issue
for i = 1:length(XData)
    % current is usually Feature 2; we integrate it over the 50-second window
    current_signal = XData{i}(2,:);
    integrated_current = cumsum(current_signal); 
    XData{i} = [XData{i}; integrated_current]; 
end

% Split Data
rng(42); n = length(YData); idx = randperm(n);
it = idx(1:round(0.8*n)); iv = idx(round(0.8*n)+1:round(0.9*n)); itest = idx(round(0.9*n)+1:end);

% --- 3. DEEPER ARCHITECTURE ---
layers = [
    sequenceInputLayer(4, 'MinLength', 50, 'Name', 'input') % 4 Features now
    
    % CNN Feature Extraction
    convolution1dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    % Dual LSTM for complex temporal dependencies
    lstmLayer(150, 'OutputMode', 'sequence') 
    dropoutLayer(0.2)
    lstmLayer(150, 'OutputMode', 'last')
    
    % Dense Regression Head
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];

% --- 4. FINE-TUNED TRAINING OPTIONS ---
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...            % Increased epochs
    'MiniBatchSize', 512, ...        % Smaller batch for finer weight updates
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 20, ...
    'L2Regularization', 0.001, ...   % Prevents overfitting at high epochs
    'ValidationData', {XData(iv), YData(iv)}, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% --- 5. TRAINING ---
fprintf('Training High-Precision Hybrid Model...\n');
net_95 = trainNetwork(XData(it), YData(it), layers, options);

% --- 6. EVALUATION ---
YPred = predict(net_95, XData(itest));
YTrue = YData(itest);
maeVal = mean(abs(YTrue - YPred));
accuracy = (1 - maeVal) * 100;

% --- 7. TABBED VISUALIZATION ---
hFig = figure('Name', '95% Accuracy Pursuit', 'Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
hTabGroup = uitabgroup(hFig);

% Tab 1: Predictions (Zoomed 1:300)
tab1 = uitab(hTabGroup, 'Title', 'SoC Predictions');
ax1 = axes('Parent', tab1);
zR = 1:min(300, length(YTrue)); 

subplot(2,1,1, ax1);
plot(YTrue(zR), 'b-o', 'MarkerSize', 4, 'LineWidth', 1.5); hold on;
plot(YPred(zR), 'r--x', 'MarkerSize', 4, 'LineWidth', 1.2);
ylabel('State of Charge'); title('LFP SoC Tracking');
grid on; legend('Actual', 'Predicted');

subplot(2,1,2, ax1);
plot(YTrue(zR), 'b', 'LineWidth', 2); hold on;
plot(YPred(zR), 'r--', 'LineWidth', 2);
title(['Model Accuracy: ', num2str(accuracy, '%.2f'), '%']);
grid on;

% Tab 2: Scatter Plot
tab2 = uitab(hTabGroup, 'Title', 'Scatter Analysis');
ax2 = axes('Parent', tab2);
scatter(ax2, YTrue, YPred, 10, 'filled', 'MarkerFaceAlpha', 0.1); hold on;
plot(ax2, [0 1], [0 1], 'r', 'LineWidth', 2);
xlabel('True SoC'); ylabel('Predicted SoC'); title('Regression Consistency');

fprintf('\nFinal Achieved Accuracy: %.2f%%\n', accuracy);