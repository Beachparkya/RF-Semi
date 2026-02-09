% % 声明全局变量以存储优化历史
% global optimizationHistoryTSS optimizationHistoryTN optimizationHistoryTP;

% % 初始化优化历史变量
% optimizationHistoryTSS = table([], [], [], [], 'VariableNames', {'NumTrees', 'MaxDepth', 'LeafSize', 'RMSE'});
% optimizationHistoryTN = table([], [], [], [], 'VariableNames', {'NumTrees', 'MaxDepth', 'LeafSize', 'RMSE'});
% optimizationHistoryTP = table([], [], [], [], 'VariableNames', {'NumTrees', 'MaxDepth', 'LeafSize', 'RMSE'});

% 加载数据集
load('2019data.mat');
load('2018data.mat');

% 数据预处理
X_2019 = dataset2019{:, 1:19};
Y_2019_TSS = dataset2019.TSS;
Y_2019_TN = dataset2019.TN;
Y_2019_TP = dataset2019.TP;
X_val = dataset2018{:, 1:19};
Y_val_TSS = dataset2018.TSS;
Y_val_TN = dataset2018.TN;
Y_val_TP = dataset2018.TP;


% 设置GA和交叉验证的参数
initialPopulation = [100, 15, 1];
options = optimoptions('ga', 'PopulationSize', 5, 'MaxGenerations', 5, 'Display', 'iter', 'InitialPopulationMatrix', initialPopulation);
k = 5; % K折交叉验证的折数

% 超参数优化
lb = [20, 5, 1]; % 下界
ub = [200, 50, 20]; % 上界
% [optimalParams_TSS, ~] = ga(@(params) objectiveFunction(params, X_train, Y_train_TSS, k), 3, [], [], [], [], lb, ub, [], options);
% [optimalParams_TN, ~] = ga(@(params) objectiveFunction(params, X_train, Y_train_TN, k), 3, [], [], [], [], lb, ub, [], options);
% [optimalParams_TP, ~] = ga(@(params) objectiveFunction(params, X_train, Y_train_TP, k), 3, [], [], [], [], lb, ub, [], options);
[optimalParams_TSS, ~] = ga(@(params) objectiveFunction(params, X_2019, Y_2019_TSS, k), 3, [], [], [], [], lb, ub, [], options);
[optimalParams_TN, ~] = ga(@(params) objectiveFunction(params, X_2019, Y_2019_TN, k), 3, [], [], [], [], lb, ub, [], options);
[optimalParams_TP, ~] = ga(@(params) objectiveFunction(params, X_2019, Y_2019_TP, k), 3, [], [], [], [], lb, ub, [], options);

% Randomly split the 2019 dataset into training and testing sets (80%-20%)
cv = cvpartition(size(X_2019, 1), 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);

X_train = X_2019(idxTrain, :);
Y_train_TSS = Y_2019_TSS(idxTrain);
Y_train_TN = Y_2019_TN(idxTrain);
Y_train_TP = Y_2019_TP(idxTrain);

X_test = X_2019(idxTest, :);
Y_test_TSS = Y_2019_TSS(idxTest);
Y_test_TN = Y_2019_TN(idxTest);
Y_test_TP = Y_2019_TP(idxTest);

% 重新训练模型
finalModel_TSS = trainRF(optimalParams_TSS, X_train, Y_train_TSS);
finalModel_TN = trainRF(optimalParams_TN, X_train, Y_train_TN);
finalModel_TP = trainRF(optimalParams_TP, X_train, Y_train_TP);


% 验证集上的性能评估
[TSS_val_rmse, TSS_val_nse] = evaluatePerformance(finalModel_TSS, X_val, Y_val_TSS, 'TSS');
[TN_val_rmse, TN_val_nse] = evaluatePerformance(finalModel_TN, X_val, Y_val_TN, 'TN');
[TP_val_rmse, TP_val_nse] = evaluatePerformance(finalModel_TP, X_val, Y_val_TP, 'TP');


% % Combine the optimal parameters into a table
% % 将最优参数组合成一个表
% optimalHyperparameters = table([optimalParams_TSS(1); optimalParams_TN(1); optimalParams_TP(1)], ...
%                                [optimalParams_TSS(2); optimalParams_TN(2); optimalParams_TP(2)], ...
%                                [optimalParams_TSS(3); optimalParams_TN(3); optimalParams_TP(3)], ...
%                                'VariableNames', {'NumTrees', 'MaxDepth', 'LeafSize'}, ...
%                                'RowNames', {'TSS', 'TN', 'TP'});
% 
% % Write the table to an Excel file
% writetable(optimalHyperparameters, 'OptimalHyperparameters.xlsx', 'WriteRowNames', true);


% Initialize tables to store RMSE and NSE for training, testing, and validation datasets
performanceMetrics = struct();
performanceMetrics.TSS = table([], [], [], 'VariableNames', {'RMSE_Train', 'RMSE_Test', 'RMSE_Val'});
performanceMetrics.TN = table([], [], [], 'VariableNames', {'RMSE_Train', 'RMSE_Test', 'RMSE_Val'});
performanceMetrics.TP = table([], [], [], 'VariableNames', {'RMSE_Train', 'RMSE_Test', 'RMSE_Val'});

% Evaluate performance on validation datasets and store predicted values
[TSS_val_rmse, TSS_val_nse, Y_pred_TSS_val] = evaluatePerformance(finalModel_TSS, X_val, Y_val_TSS, 'TSS');
[TN_val_rmse, TN_val_nse, Y_pred_TN_val] = evaluatePerformance(finalModel_TN, X_val, Y_val_TN, 'TN');
[TP_val_rmse, TP_val_nse, Y_pred_TP_val] = evaluatePerformance(finalModel_TP, X_val, Y_val_TP, 'TP');

% Create tables to store the predicted values
predictedValues = table(Y_val_TSS, Y_pred_TSS_val, Y_val_TN, Y_pred_TN_val, Y_val_TP, Y_pred_TP_val, 'VariableNames', {'Actual_TSS', 'Predicted_TSS', 'Actual_TN', 'Predicted_TN', 'Actual_TP', 'Predicted_TP'});

% Optionally, write the table to an Excel file
writetable(predictedValues, 'PredictedValues.xlsx');

% Evaluate performance on training, testing, and validation datasets
% For TSS
[TSS_rmse_train, TSS_nse_train] = evaluatePerformance(finalModel_TSS, X_train, Y_train_TSS, 'TSS');
[TSS_rmse_test, TSS_nse_test] = evaluatePerformance(finalModel_TSS, X_test, Y_test_TSS, 'TSS');
[TSS_rmse_val, TSS_nse_val] = evaluatePerformance(finalModel_TSS, X_val, Y_val_TSS, 'TSS');

% Store the performance metrics for TSS
performanceMetrics.TSS = [performanceMetrics.TSS; {TSS_rmse_train, TSS_rmse_test, TSS_rmse_val}];

% Repeat the process for TN and TP
% For TN
[TN_rmse_train, TN_nse_train] = evaluatePerformance(finalModel_TN, X_train, Y_train_TN, 'TN');
[TN_rmse_test, TN_nse_test] = evaluatePerformance(finalModel_TN, X_test, Y_test_TN, 'TN');
[TN_rmse_val, TN_nse_val] = evaluatePerformance(finalModel_TN, X_val, Y_val_TN, 'TN');

% Store the performance metrics for TN
performanceMetrics.TN = [performanceMetrics.TN; {TN_rmse_train, TN_rmse_test, TN_rmse_val}];

% For TP
[TP_rmse_train, TP_nse_train] = evaluatePerformance(finalModel_TP, X_train, Y_train_TP, 'TP');
[TP_rmse_test, TP_nse_test] = evaluatePerformance(finalModel_TP, X_test, Y_test_TP, 'TP');
[TP_rmse_val, TP_nse_val] = evaluatePerformance(finalModel_TP, X_val, Y_val_TP, 'TP');

% Store the performance metrics for TP
performanceMetrics.TP = [performanceMetrics.TP; {TP_rmse_train, TP_rmse_test, TP_rmse_val}];

% % Display the performance metrics
% disp(performanceMetrics.TSS);
% disp(performanceMetrics.TN);
% disp(performanceMetrics.TP);

% Organize performance metrics into tables
performanceMetricsTSS = table([TSS_rmse_train; TSS_rmse_test; TSS_rmse_val], [TSS_nse_train; TSS_nse_test; TSS_nse_val], 'VariableNames', {'RMSE', 'NSE'}, 'RowNames', {'Train', 'Test', 'Validation'});
performanceMetricsTN = table([TN_rmse_train; TN_rmse_test; TN_rmse_val], [TN_nse_train; TN_nse_test; TN_nse_val], 'VariableNames', {'RMSE', 'NSE'}, 'RowNames', {'Train', 'Test', 'Validation'});
performanceMetricsTP = table([TP_rmse_train; TP_rmse_test; TP_rmse_val], [TP_nse_train; TP_nse_test; TP_nse_val], 'VariableNames', {'RMSE', 'NSE'}, 'RowNames', {'Train', 'Test', 'Validation'});

% Write the tables to an Excel file
writetable(performanceMetricsTSS, 'PerformanceMetricsTSS.xlsx', 'WriteRowNames', true);
writetable(performanceMetricsTN, 'PerformanceMetricsTN.xlsx', 'WriteRowNames', true);
writetable(performanceMetricsTP, 'PerformanceMetricsTP.xlsx', 'WriteRowNames', true);

% 定义函数
function rmse_avg = objectiveFunction(params, X, Y, k)
%     global optimizationHistoryTSS optimizationHistoryTN optimizationHistoryTP;
    cvp = cvpartition(size(X, 1), 'KFold', k);
    rmse_values = zeros(cvp.NumTestSets, 1);

    for i = 1:cvp.NumTestSets
        idxTrain = training(cvp, i);
        idxTest = test(cvp, i);

        X_train = X(idxTrain, :);
        Y_train = Y(idxTrain);
        X_test = X(idxTest, :);
        Y_test = Y(idxTest);

        % 训练模型
        rf = trainRF(params, X_train, Y_train);

        % 预测和计算RMSE
        Y_pred = predict(rf, X_test);
        rmse_values(i) = sqrt(mean((Y_test - Y_pred).^2));
    end

    % 计算平均RMSE
    rmse_avg = mean(rmse_values);

    % 更新优化历史
%     tempTable = table(params(1), params(2), params(3), rmse_avg, 'VariableNames', {'NumTrees', 'MaxDepth', 'LeafSize', 'RMSE'});
%     optimizationHistoryTSS = [optimizationHistoryTSS; tempTable];
end

function rf = trainRF(params, X, Y)
    numTreesOpt = round(params(1));
    maxDepthOpt = round(params(2));
    leafSizeOpt = round(params(3));
    rf = TreeBagger(numTreesOpt, X, Y, 'OOBPredictorImportance', 'On', 'Method', 'regression', 'MaxNumSplits', maxDepthOpt, 'MinLeafSize', leafSizeOpt);
end


% Modify the evaluatePerformance function to return predictions
function [rmse, nse, Y_pred] = evaluatePerformance(rf, X, Y, label)
    Y_pred = predict(rf, X);
    rmse = sqrt(mean((Y - Y_pred).^2));
    nse = 1 - sum((Y - Y_pred).^2) / sum((Y - mean(Y)).^2);
    fprintf('%s RMSE: %f, NSE: %f\n', label, rmse, nse);
end
% 
% function plotOptimizationHistory(optimizationHistory, targetVar)
%     figure;
%     subplot(3,1,1);
%     plot(optimizationHistory.NumTrees, optimizationHistory.RMSE, 'b.');
%     xlabel('Number of Trees');
%     ylabel('RMSE');
%     title(['RMSE vs Number of Trees for ', targetVar]);
% 
%     subplot(3,1,2);
%     plot(optimizationHistory.MaxDepth, optimizationHistory.RMSE, 'r.');
%     xlabel('Maximum Depth');
%     ylabel('RMSE');
%     title(['RMSE vs Maximum Depth for ', targetVar]);
% 
%     subplot(3,1,3);
%     plot(optimizationHistory.LeafSize, optimizationHistory.RMSE, 'g.');
%     xlabel('Minimum Leaf Size');
%     ylabel('RMSE');
%     title(['RMSE vs Minimum Leaf Size for ', targetVar]);
% end
