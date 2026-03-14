%% NASA Battery Dataset Preprocessing for Deep Learning
clear; clc; close all;

% --- Configuration ---
basePath = 'C:\Matlab Projects\bms\Raw NMC';
dataFolder = fullfile(basePath, 'data');
metadata = readtable(fullfile(basePath, 'metadata.csv'));

fixedTimeSteps = 200; % Number of time steps per discharge cycle (Standardized)
featureNames = {'Voltage_measured', 'Current_measured', 'Temperature_measured'};
numFeatures = length(featureNames);

% --- 1. Filter for Discharge Cycles only ---
% Note: Capacity is only recorded during discharge.
dischargeMeta = metadata(strcmp(metadata.type, 'discharge'), :);
numSamples = height(dischargeMeta);

% Initialize 3D Arrays: [Samples x TimeSteps x Features]
X = zeros(numSamples, fixedTimeSteps, numFeatures);
Y = zeros(numSamples, 1); % Target: Capacity

fprintf('Starting Preprocessing of %d discharge cycles...\n', numSamples);

% --- 2. Main Processing Loop ---
for i = 1:numSamples
    % Load the specific CSV file
    fileName = dischargeMeta.filename{i};
    filePath = fullfile(dataFolder, fileName);
    
    if exist(filePath, 'file')
        rawFileData = readtable(filePath);
        
        % Extract specific features
        cycleData = rawFileData{:, featureNames};
        
        % --- 3. Handle Variable Lengths (Interpolation) ---
        % Deep learning models need fixed input sizes. We interpolate to 'fixedTimeSteps'.
        oldSteps = size(cycleData, 1);
        newSteps = linspace(1, oldSteps, fixedTimeSteps);
        
        for f = 1:numFeatures
            X(i, :, f) = interp1(1:oldSteps, cycleData(:, f), newSteps, 'linear');
        end
        
        % --- 4. Store Target Variable ---
        Y(i) = dischargeMeta.Capacity(i);
    end
    
    if mod(i, 50) == 0, fprintf('Processed %d/%d files...\n', i, numSamples); end
end

% --- 5. Normalization ---
% Scale features between 0 and 1 for faster convergence
for f = 1:numFeatures
    featMin = min(X(:, :, f), [], 'all');
    featMax = max(X(:, :, f), [], 'all');
    X(:, :, f) = (X(:, :, f) - featMin) / (featMax - featMin);
end

% Target normalization (Optional but recommended)
capMin = min(Y); capMax = max(Y);
Y_scaled = (Y - capMin) / (capMax - capMin);

% --- 6. Train/Test Split (80/20) ---
idx = randperm(numSamples);
splitIdx = round(0.8 * numSamples);

trainX = X(idx(1:splitIdx), :, :);
trainY = Y_scaled(idx(1:splitIdx));

testX = X(idx(splitIdx+1:end), :, :);
testY = Y_scaled(idx(splitIdx+1:end));

fprintf('\nPreprocessing Complete!\n');
fprintf('Training Shape: [%s]\n', num2str(size(trainX)));
fprintf('Ready for CNN, LSTM, or CNN-LSTM models.\n');

% Save processed data for the next step
save('PreprocessedBatteryData.mat', 'trainX', 'trainY', 'testX', 'testY', 'capMin', 'capMax');