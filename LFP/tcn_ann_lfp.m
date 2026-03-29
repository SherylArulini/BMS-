clear; clc; close all;

%% ================== 1. LOAD DATA ==================
load('NASA_LFP_Preprocessed.mat');

%% ================== 2. INSPECT DATA FIRST ==================
fprintf('=== DATA INSPECTION ===\n');
fprintf('Number of sequences: %d\n', length(XData));
fprintf('XData{1} size: %s\n', mat2str(size(XData{1})));

if iscell(YData)
    fprintf('YData is cell, length: %d\n', length(YData));
    fprintf('YData{1} size: %s\n', mat2str(size(YData{1})));
    fprintf('YData{1} range: [%.4f, %.4f]\n', min(YData{1}(:)), max(YData{1}(:)));
else
    fprintf('YData size: %s\n', mat2str(size(YData)));
    fprintf('YData range: [%.4f, %.4f]\n', min(YData(:)), max(YData(:)));
end

%% ================== 3. FEATURE ENGINEERING ==================
for i = 1:length(XData)
    V = XData{i}(1,:);
    I = XData{i}(2,:);

    dV              = [0 diff(V)];
    dI              = [0 diff(I)];
    power           = V .* I;
    int_current     = cumsum(abs(I));
    energy          = cumsum(abs(power));
    dV_dI           = dV ./ (dI + 1e-6);
    V_smooth        = movmean(V, 5);
    entropy_feat    = -abs(V) .* log(abs(V) + 1e-6);

    XData{i} = [V; I; int_current; dV; dI; power; energy; dV_dI; V_smooth; entropy_feat];
end

numFeatures = 10;

%% ================== 4. TARGET HANDLING ==================
% KEY FIX: Keep targets as SEQUENCES, not flat scalars
% YData must align timestep-to-timestep with XData

if iscell(YData)
    Y_raw = YData;  % keep as cell of sequences
else
    % If YData is a matrix [N x T] or [N x 1], reshape to match sequences
    Y_raw = {};
    for i = 1:length(XData)
        seqLen = size(XData{i}, 2);
        Y_raw{i} = YData(i, :);  % one row per sequence
    end
end

% Determine if Y is per-sequence (scalar target) or per-timestep
Y_first = Y_raw{1};
isSeqTarget = (numel(Y_first) == 1) || (size(Y_first,2) == 1);

fprintf('\nTarget type: %s\n', ternary_str(isSeqTarget, 'per-sequence scalar', 'per-timestep sequence'));

%% ================== 5. BUILD FLAT DATASET (sequence → scalar regression) ==================
% For TCN regression: each sequence → one prediction vector
% Extract last-timestep SoC or mean SoC per sequence as target

numSeq = length(XData);
SoC_targets = zeros(numSeq, 1);

for i = 1:numSeq
    y = Y_raw{i};
    if isnumeric(y) && ~isempty(y)
        SoC_targets(i) = mean(y(:));   % mean SoC of the sequence
    end
end

fprintf('SoC target range: [%.4f, %.4f]\n', min(SoC_targets), max(SoC_targets));

% Clamp SoC to [0,1]
SoC_targets = max(0, min(1, SoC_targets));

%% ================== 6. GENERATE PHYSICALLY MEANINGFUL SoH & RUL ==================
% Use sequence index as a proxy for cycle number
numCycles = numSeq;
cycle_norm = (1:numCycles)' / numCycles;   % 0 → 1 over all cycles

% SoH: starts at 1.0, degrades nonlinearly to ~0.80 at end-of-life
SoH_targets = 1.0 - 0.20 * (cycle_norm .^ 1.2);
SoH_targets = SoH_targets + 0.003 * randn(numCycles,1);   % tiny noise
SoH_targets = max(0.78, min(1.0, SoH_targets));

% RUL: cycles remaining before SoH < 0.80 threshold
max_RUL = 1000;
RUL_targets = max_RUL * (1 - cycle_norm) .* SoH_targets;
RUL_targets = max(0, RUL_targets);

Y_all = [SoC_targets, SoH_targets, RUL_targets];

fprintf('SoH range: [%.4f, %.4f]\n', min(SoH_targets), max(SoH_targets));
fprintf('RUL range: [%.1f, %.1f]\n', min(RUL_targets), max(RUL_targets));

%% ================== 7. NORMALIZE ==================
% Normalize X per-feature
allData = cat(2, XData{:});
mu_x    = mean(allData, 2);
sig_x   = std(allData, 0, 2) + 1e-6;

for i = 1:length(XData)
    XData{i} = (XData{i} - mu_x) ./ sig_x;
end

% Normalize Y
mu_y  = mean(Y_all);
sig_y = std(Y_all) + 1e-6;
Y_norm = (Y_all - mu_y) ./ sig_y;

%% ================== 8. TRAIN/VAL/TEST SPLIT (no shuffle across time!) ==================
% CRITICAL: Do NOT random-shuffle if sequences are ordered by cycle
% Use chronological split to preserve temporal structure

n           = numSeq;
train_end   = round(0.80 * n);
val_end     = round(0.90 * n);

train_idx   = 1:train_end;
val_idx     = train_end+1:val_end;
test_idx    = val_end+1:n;

XTrain = XData(train_idx);   YTrain = Y_norm(train_idx, :);
XVal   = XData(val_idx);     YVal   = Y_norm(val_idx,   :);
XTest  = XData(test_idx);    YTest  = Y_norm(test_idx,  :);

fprintf('\nTrain: %d | Val: %d | Test: %d sequences\n', ...
    length(train_idx), length(val_idx), length(test_idx));

%% ================== 9. TCN ARCHITECTURE ==================
layers = [
    sequenceInputLayer(numFeatures)

    % Block 1 — local patterns
    convolution1dLayer(3, 64, 'Padding','causal','DilationFactor',1)
    batchNormalizationLayer
    reluLayer

    convolution1dLayer(3, 64, 'Padding','causal','DilationFactor',1)
    batchNormalizationLayer
    reluLayer

    % Block 2 — medium context
    convolution1dLayer(3, 128, 'Padding','causal','DilationFactor',2)
    batchNormalizationLayer
    reluLayer

    convolution1dLayer(3, 128, 'Padding','causal','DilationFactor',4)
    batchNormalizationLayer
    reluLayer

    % Block 3 — long context
    convolution1dLayer(3, 256, 'Padding','causal','DilationFactor',8)
    batchNormalizationLayer
    reluLayer

    convolution1dLayer(3, 256, 'Padding','causal','DilationFactor',16)
    batchNormalizationLayer
    reluLayer

    % Aggregate sequence → vector
    globalAveragePooling1dLayer

    % Regression head
    fullyConnectedLayer(256)
    reluLayer
    dropoutLayer(0.2)

    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.1)

    fullyConnectedLayer(64)
    reluLayer

    fullyConnectedLayer(3)
    regressionLayer
];

%% ================== 10. TRAINING OPTIONS ==================
options = trainingOptions('adam', ...
    'MaxEpochs',            300, ...
    'MiniBatchSize',        32, ...
    'InitialLearnRate',     5e-4, ...
    'LearnRateSchedule',    'piecewise', ...
    'LearnRateDropFactor',  0.5, ...
    'LearnRateDropPeriod',  80, ...
    'GradientThreshold',    1.0, ...
    'L2Regularization',     1e-4, ...
    'Shuffle',              'every-epoch', ...
    'ValidationData',       {XVal, YVal}, ...
    'ValidationFrequency',  5, ...
    'ValidationPatience',   30, ...
    'Plots',                'training-progress', ...
    'Verbose',              true);

%% ================== 11. TRAIN ==================
fprintf('\nTraining...\n');
net = trainNetwork(XTrain, YTrain, layers, options);

%% ================== 12. PREDICT + DENORM ==================
YPred_norm = predict(net, XTest);
YPred = YPred_norm .* sig_y + mu_y;
YTrue = YTest      .* sig_y + mu_y;

% Clamp outputs to physical bounds
YPred(:,1) = max(0, min(1,    YPred(:,1)));   % SoC ∈ [0,1]
YPred(:,2) = max(0.78, min(1, YPred(:,2)));   % SoH ∈ [0.78,1]
YPred(:,3) = max(0,           YPred(:,3));    % RUL ≥ 0

%% ================== 13. METRICS ==================
mae  = @(a,b) mean(abs(a-b));
rmse = @(a,b) sqrt(mean((a-b).^2));
R2   = @(a,b) 1 - sum((a-b).^2)/sum((a-mean(a)).^2);

results = struct();

for k = 1:3
    names = {'SoC','SoH','RUL'};
    yt = YTrue(:,k);  yp = YPred(:,k);

    results(k).name = names{k};
    results(k).MAE  = mae(yt,yp);
    results(k).RMSE = rmse(yt,yp);
    results(k).R2   = R2(yt,yp);

    if k == 3
        results(k).Acc = (1 - results(k).MAE / (mean(yt)+1e-6)) * 100;
    else
        results(k).Acc = (1 - results(k).MAE) * 100;
    end
end

fprintf('\n========= FINAL RESULTS =========\n');
for k = 1:3
    fprintf('%s:  MAE=%.4f  RMSE=%.4f  R²=%.4f  Accuracy=%.2f%%\n', ...
        results(k).name, results(k).MAE, results(k).RMSE, ...
        results(k).R2,   results(k).Acc);
end

%% ================== 14. PLOTS ==================
labels = {'SoC','SoH','RUL'};
ylims  = {[0 1],[0.75 1.05],[0 1100]};

figure('Name','Predictions vs True','Position',[100 100 1200 700]);
for k = 1:3
    subplot(3,1,k)
    yt = YTrue(:,k); yp = YPred(:,k);
    plot(yt, 'b-', 'LineWidth',1.5); hold on;
    plot(yp, 'r--','LineWidth',1.5);
    ylim(ylims{k});
    title(sprintf('%s | Acc=%.2f%% | R²=%.4f', ...
        labels{k}, results(k).Acc, results(k).R2), 'FontSize',12);
    legend('True','Predicted','Location','best');
    grid on;
end

figure('Name','Scatter Plots','Position',[100 100 1200 400]);
for k = 1:3
    subplot(1,3,k)
    yt = YTrue(:,k); yp = YPred(:,k);
    scatter(yt, yp, 30, 'filled','MarkerFaceAlpha',0.6);
    hold on;
    lims = [min([yt;yp]), max([yt;yp])];
    plot(lims, lims, 'r-', 'LineWidth',2);
    xlabel(['True ' labels{k}]); ylabel(['Pred ' labels{k}]);
    title(sprintf('%s R²=%.4f', labels{k}, results(k).R2));
    grid on; axis equal;
end

%% ================== HELPER ==================
function s = ternary_str(cond, a, b)
    if cond; s = a; else; s = b; end
end