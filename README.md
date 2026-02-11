# RF-Semi
semi-supervised RF model
This repository contains the data, MATLAB code, and supporting output files for A a semi-supervised machine learning framework to predict urban stormwater runoff quality in data-deficient areas.
The workflow implements a semi-supervised Random Forest regression framework for predicting event mean concentrations (EMCs) of TSS, TN, and TP using:
(1)Labeled events: rainfall + catchment features with observed EMCs
(2)Unlabeled events: rainfall + catchment features without EMCs
(3)Pseudo-labeling: generate pseudo-labels from the best supervised model and filter them using an ensemble agreement criterion (CV)
(4)Semi-supervised retraining: train RF-Semi with labeled + high-confidence pseudo-labeled events
1. Repository contents
From the repository (as shown in the GitHub snapshot), the key files are:
(1)MATLAB code: RF_Simi.m
(2)Data:2018data.mat(Event-level dataset for testing);2019data.mat(Event-level dataset for training/validation);Raw Data.xlsx(Raw event table used to assemble the MATLAB datasets)
(3)Hyperparameter optimization & results:optimizationHistoryTSS.csv(Genetic algorithm (GA) search history for each target);OptimalHyperparameters.xlsx(Best hyperparameter settings);bestRFModel.mat(Saved best semi-supervised RF model)
(4)Performance & pseudo-label selection:PerformanceMetricsTSS.xlsx(Metrics summary (e.g., RMSE, NSE) across train/val/test);pseudo-label selection.xlsx(CV-based pseudo-label filtering results)
2. Method overview
The framework contains three major stages:
Stage A — Supervised RF model development (RF-M1 / RF-M2 / RF-M3)
Three supervised configurations are trained to quantify the effect of data/feature enrichment:
RF-M1: rainfall features only, single catchment baseline
RF-M2: rainfall features only, multi-catchment
RF-M3: rainfall + catchment features, multi-catchment (selected as best base model)
Hyperparameters are tuned using Genetic Algorithm + 5-fold CV on the 2019 dataset.
Stage B — Pseudo-label generation with ensemble agreement filtering
Using the best supervised model (typically RF-M3), pseudo-labels are generated for unlabeled events.
To control pseudo-label quality, an ensemble of 27 RF variants is created by perturbing the tuned hyperparameters around the optimum; each variant predicts the unlabeled events. Confidence is quantified via Coefficient of Variation (CV) across the 27 predictions.Events with CV below a threshold (per pollutant) are retained as pseudo-labeled samples.
Stage C — Semi-supervised retraining (RF-Semi)
Combine:original labeled events, andselected high-confidence pseudo-labeled events.
Then retrain RF with the same hyperparameters as the best supervised model to isolate the effect of data enrichment. Final evaluation is performed on the independent 2018 test set.
Notes on Model Consistency：
It is important to clarify that:
RF-M1, RF-M2, RF-M3, and RF-Semi share exactly the same Random Forest structure, hyperparameter optimization procedure, and training/validation strategy.There is no structural or algorithmic difference between these models.
The differences arise solely from:Feature composition (rainfall-only vs rainfall + catchment features)；Training dataset composition (labeled only vs labeled + pseudo-labeled)；Therefore, the core methodological contribution lies in:The structured handling and enrichment of training data，rather than modifications to the machine learning algorithm itself.As a result, the model code remains fundamentally identical across configurations, and performance differences reflect changes in training data structure.
3. Requirements
Software:MATLAB (R2020b+ recommended; earlier versions may work);Required toolboxes:Statistics and Machine Learning Toolbox (Random Forest / TreeBagger),Global Optimization Toolbox (for GA hyperparameter tuning; if GA is already done, you can skip this part).
