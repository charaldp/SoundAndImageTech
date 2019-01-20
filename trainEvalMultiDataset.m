%% FIS Model Paramemters
NF = 14;
NR = 12;
K_folds = 4; % 4 folds is an optimal number for parallel processing using 4 cores processor
clust_type = 1; % Use Fuzzy C-Means FIS initialization

dataGTZAN = csvread('Output/afterProcNoAttrsGTZAN.csv');

dataMusan = csvread('Output/afterProcNoAttrsMusan.csv');

dataMirex = csvread('Output/afterProcNoAttrsMirex.csv');

[accGTZANonTrain,accFoldsGTZAN,TSK_models_GTZAN,rankGTZAN] = classificationFuzzy(dataGTZAN,NF,NR,K_folds,clust_type);

[accMusanOnTrain,accFoldsMusan,TSK_models_Musan,rankMusan] = classificationFuzzy(dataMusan,NF,NR,K_folds,clust_type);

%% Evaluate Trained Models on "Unknown" Data
OA1 = zeros(1,K_folds);
OA2 = zeros(1,K_folds);
OA3 = zeros(1,K_folds);
OA4 = zeros(1,K_folds);
ConfusionMatrixGTZAN_Musan = zeros(2,2,K_folds);
ConfusionMatrixGTZAN_Mirex = zeros(2,2,K_folds);
ConfusionMatrixMusan_GTZAN = zeros(2,2,K_folds);
ConfusionMatrixMusan_Mirex = zeros(2,2,K_folds);

for k = 1:K_folds
    predGTZAN_Musan = evalfis(dataMusan(:,rankGTZAN(1:NF)), TSK_models_GTZAN(k));
    predGTZAN_Mirex = evalfis(dataMirex(:,rankGTZAN(1:NF)), TSK_models_GTZAN(k));
    predMusan_GTZAN = evalfis(dataGTZAN(:,rankMusan(1:NF)), TSK_models_Musan(k));
    predMusan_Mirex = evalfis(dataMirex(:,rankMusan(1:NF)), TSK_models_Musan(k));
    
    predGTZAN_Musan = min(max(round(predGTZAN_Musan),0),1);
    predGTZAN_Mirex = min(max(round(predGTZAN_Mirex),0),1);
    predMusan_GTZAN = min(max(round(predMusan_GTZAN),0),1);
    predMusan_Mirex = min(max(round(predMusan_Mirex),0),1);
    
    actualGTZAN = dataGTZAN(:,23);
    actualMusan = dataMusan(:,23);
    actualMirex = dataMirex(:,23);
    
    for l=1:length(actualGTZAN)
        ConfusionMatrixMusan_GTZAN(predMusan_GTZAN(l)+1,actualGTZAN(l)+1,k) = ConfusionMatrixMusan_GTZAN(predMusan_GTZAN(l)+1,actualGTZAN(l)+1,k) + 1;
    end
    for l=1:length(actualMusan)
        ConfusionMatrixGTZAN_Musan(predGTZAN_Musan(l)+1,actualMusan(l)+1,k) = ConfusionMatrixGTZAN_Musan(predGTZAN_Musan(l)+1,actualMusan(l)+1,k) + 1;
    end
    for l=1:length(actualMirex)
        ConfusionMatrixGTZAN_Mirex(predGTZAN_Mirex(l)+1,actualMirex(l)+1,k) = ConfusionMatrixGTZAN_Mirex(predGTZAN_Mirex(l)+1,actualMirex(l)+1,k) + 1;
        ConfusionMatrixMusan_Mirex(predMusan_Mirex(l)+1,actualMirex(l)+1,k) = ConfusionMatrixMusan_Mirex(predMusan_Mirex(l)+1,actualMirex(l)+1,k) + 1;
    end
    % Model Evaluation Metrics
    OA1(k) = sum(sum(eye(2) .* ConfusionMatrixGTZAN_Musan(:,:,k)))/sum(sum(ConfusionMatrixGTZAN_Musan(:,:,k)));
    OA2(k) = sum(sum(eye(2) .* ConfusionMatrixGTZAN_Mirex(:,:,k)))/sum(sum(ConfusionMatrixGTZAN_Mirex(:,:,k)));
    OA3(k) = sum(sum(eye(2) .* ConfusionMatrixMusan_GTZAN(:,:,k)))/sum(sum(ConfusionMatrixMusan_GTZAN(:,:,k)));
    OA4(k) = sum(sum(eye(2) .* ConfusionMatrixMusan_Mirex(:,:,k)))/sum(sum(ConfusionMatrixMusan_Mirex(:,:,k)));
end
meanConfusionMatrix1 = sum(ConfusionMatrixGTZAN_Musan,3)/K_folds;
meanConfusionMatrix2 = sum(ConfusionMatrixGTZAN_Mirex,3)/K_folds;
meanConfusionMatrix3 = sum(ConfusionMatrixMusan_GTZAN,3)/K_folds;
meanConfusionMatrix4 = sum(ConfusionMatrixMusan_Mirex,3)/K_folds;

fprintf('ConfusionMatrixGTZAN_Musan = \n%6.f\t%6.f\n%6.f\t%6.f\n',meanConfusionMatrix1);
fprintf('ConfusionMatrixGTZAN_Mirex = \n%6.f\t%6.f\n%6.f\t%6.f\n',meanConfusionMatrix2);
fprintf('ConfusionMatrixMusan_GTZAN = \n%6.f\t%6.f\n%6.f\t%6.f\n',meanConfusionMatrix3);
fprintf('ConfusionMatrixMusan_Mirex = \n%6.f\t%6.f\n%6.f\t%6.f\n',meanConfusionMatrix4);

accuracyGTZAN_Musan = mean(OA1);
accuracyGTZAN_Mirex = mean(OA2);
accuracyMusan_GTZAN = mean(OA3);
accuracyMusan_Mirex = mean(OA4);
