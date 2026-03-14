%% LSTM MODEL FOR NMC BATTERY SOC and CAPACITY ESTIMATION
clear; clc; close all;

rng(42)
%% 1. DATA LOADING
basePath = 'C:\Matlab Projects\bms\Raw NMC';
dataFolder = fullfile(basePath,'data');
metadata = readtable(fullfile(basePath,'metadata.csv'));

fixedTimeSteps = 200;
featureNames = {'Voltage_measured','Current_measured','Temperature_measured'};

dischargeMeta = metadata(strcmp(metadata.type,'discharge'),:);
numSamples = height(dischargeMeta);

X_raw = zeros(numSamples,fixedTimeSteps,3);
Y_raw = zeros(numSamples,1);

fprintf('Preprocessing dataset...\n')

for i = 1:numSamples
    
    filePath = fullfile(dataFolder,dischargeMeta.filename{i});
    
    if exist(filePath,'file')
        
        data = readtable(filePath);
        
        rawS = data{:,featureNames};
        
        % Noise filtering
        rawS = movmean(rawS,5);
        
        oldSteps = size(rawS,1);
        newSteps = linspace(1,oldSteps,fixedTimeSteps);
        
        for f = 1:3
            X_raw(i,:,f) = interp1(1:oldSteps,rawS(:,f),newSteps,'linear');
        end
        
        Y_raw(i) = dischargeMeta.Capacity(i);
        
    end
end


%% 2. CLEAN DATA
nanIdx = isnan(Y_raw) | any(any(isnan(X_raw),2),3);

X_raw(nanIdx,:,:) = [];
Y_raw(nanIdx) = [];


%% 3. NORMALIZATION

X_norm = X_raw;

for f = 1:3
    
    fMin = min(X_norm(:,:,f),[],'all');
    fMax = max(X_norm(:,:,f),[],'all');
    
    X_norm(:,:,f) = (X_norm(:,:,f)-fMin)/(fMax-fMin);
    
end

capMin = min(Y_raw);
capMax = max(Y_raw);

Y_scaled = (Y_raw-capMin)/(capMax-capMin);


%% 4. DATA SPLIT

n = length(Y_raw);

idx = randperm(n);

idxTrain = idx(1:round(0.8*n));
idxVal   = idx(round(0.8*n)+1:round(0.9*n));
idxTest  = idx(round(0.9*n)+1:end);

toCell = @(data) arrayfun(@(i) squeeze(data(i,:,:))',1:size(data,1),'UniformOutput',false)';

XTrain = toCell(X_norm(idxTrain,:,:));
YTrain = Y_scaled(idxTrain);

XVal   = toCell(X_norm(idxVal,:,:));
YVal   = Y_scaled(idxVal);

XTest  = toCell(X_norm(idxTest,:,:));
YTest  = Y_scaled(idxTest);

YTrue_Ah = Y_raw(idxTest);


%% 5. LSTM ARCHITECTURE

layers = [

sequenceInputLayer(3,'MinLength',200)

lstmLayer(150,'OutputMode','sequence')
dropoutLayer(0.2)

lstmLayer(100,'OutputMode','last')
dropoutLayer(0.3)

fullyConnectedLayer(64)
reluLayer

fullyConnectedLayer(1)

regressionLayer];


%% 6. TRAINING OPTIONS (Same as other models)

options = trainingOptions('adam', ...
    'MaxEpochs',250,...
    'MiniBatchSize',32,...
    'InitialLearnRate',1e-3,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.2,...
    'LearnRateDropPeriod',100,...
    'L2Regularization',0.0005,...
    'ValidationData',{XVal,YVal},...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'Verbose',false);


fprintf('\nTraining LSTM model...\n')

netLSTM = trainNetwork(XTrain,YTrain,layers,options);


%% 7. PREDICTION

YPred_norm = predict(netLSTM,XTest);

YPred_Ah = YPred_norm*(capMax-capMin)+capMin;


%% 8. SOC CALCULATION

ratedCap = max(Y_raw);

SoC_true = YTrue_Ah/ratedCap;
SoC_pred = YPred_Ah/ratedCap;


validIdx = SoC_true > 0.05;

accuracy = (1-mean(abs(SoC_true(validIdx)-SoC_pred(validIdx))./SoC_true(validIdx)))*100;


%% 9. METRICS

rmse = sqrt(mean((YTrue_Ah-YPred_Ah).^2));
mae  = mean(abs(YTrue_Ah-YPred_Ah));


fprintf('\n=========== LSTM RESULTS ===========\n')
fprintf('Accuracy : %.2f %%\n',accuracy)
fprintf('RMSE     : %.4f Ah\n',rmse)
fprintf('MAE      : %.4f Ah\n',mae)


%% 10. PLOT

figure

plot(YTrue_Ah,'b','LineWidth',1.5)
hold on
plot(YPred_Ah,'r--','LineWidth',1.5)

legend('True Capacity','LSTM Prediction')

title(['LSTM Battery Capacity Prediction (Accuracy ',num2str(accuracy,'%.2f'),'%)'])

xlabel('Test Samples')
ylabel('Capacity (Ah)')
grid on


figure

plot(SoC_true,'b','LineWidth',1.5)
hold on
plot(SoC_pred,'r--','LineWidth',1.5)

legend('True SoC','LSTM Predicted SoC')

title(['LSTM SoC Prediction (Accuracy ',num2str(accuracy,'%.2f'),'%)'])

xlabel('Test Samples')
ylabel('State of Charge (SoC)')
ylim([0 1])
grid on

%% 11. SAVE MODEL

save('LSTM_Battery_Model.mat','netLSTM')

%% EXTRA ANALYSIS PLOTS (Added)

errors = SoC_true - SoC_pred;

rmse_ah = sqrt(mean((YTrue_Ah-YPred_Ah).^2));
mae_ah  = mean(abs(YTrue_Ah-YPred_Ah));

fprintf('RMSE Ah  : %.6f\n',rmse_ah)
fprintf('MAE  Ah  : %.6f\n',mae_ah)

% ---- Regression Scatter Plot ----
figure
scatter(SoC_true,SoC_pred,20,'filled')
hold on
plot([0 1],[0 1],'r','LineWidth',2)
xlabel('True SoC')
ylabel('Predicted SoC')
title('LSTM Regression Scatter Plot')
grid on

% ---- Error Histogram ----
figure
histogram(errors,30)
xlabel('Prediction Error')
ylabel('Frequency')
title('LSTM Error Distribution')
grid on

% ---- Error vs Samples ----
figure
plot(errors,'LineWidth',1.5)
xlabel('Test Samples')
ylabel('Error')
title('LSTM Error vs Sample Index')
grid on

