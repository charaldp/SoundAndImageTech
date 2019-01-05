%% Initialize
close all;
clear all;

TimestampData = csvread('C:\Users\Παναγιώτης\THMMY\5ο Έτος\9ο Εξάμηνο\Τεχνολογία Ήχου και Εικόνας\Data Sound Files\muspeak-mirex2015-detection-examples\ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.csv');
audio=miraudio('C:\Users\Παναγιώτης\THMMY\5ο Έτος\9ο Εξάμηνο\Τεχνολογία Ήχου και Εικόνας\Data Sound Files\muspeak-mirex2015-detection-examples\ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.mp3');
window=0.5;
framedAudio = mirframe(audio,'Length',window,'s');
framedAudioSpectrum = mirspectrum(framedAudio);
framedAudioMelSpectrum = mirspectrum(framedAudio,'Mel');


%bagpipeSpecPeaks = mirpeaks(audioSpectrum);
%bagpipeSpecPitch = mirpitch(audioSpectrum,'Mono');
 %% Framed Features
framed_rms_energy = mirrms(framedAudio);
framed_spectral_roll_off_85 = mirrolloff(framedAudioSpectrum,'Threshold',85);
framed_spectral_roll_off_90 = mirrolloff(framedAudioSpectrum,'Threshold',90);
framed_MFCCs = mirmfcc(framedAudioMelSpectrum);
framed_zcr = mirzerocross(framedAudio);
framed_spectral_flatness = mirflatness(framedAudioSpectrum);
framed_spectral_kurtosis = mirkurtosis(framedAudioSpectrum);
framed_spectral_brightness = mirbrightness(framedAudioSpectrum);
framed_spectral_irregularity = mirregularity(framedAudioSpectrum);
framed_spectral_centroid  = mircentroid(framedAudioSpectrum);

%% Extract Raw Data
%raw_pitch = mirgetdata(bagpipeSpecPitch);
raw_rms_energy = mirgetdata(framed_rms_energy);
raw_spectral_roll_off_85 = mirgetdata(framed_spectral_roll_off_85);
raw_spectral_roll_off_90 = mirgetdata(framed_spectral_roll_off_90);
raw_MFCCs = mirgetdata(framed_MFCCs);
raw_zcr = mirgetdata(framed_zcr);
raw_spectral_flatness = mirgetdata(framed_spectral_flatness);
raw_spectral_kurtosis = mirgetdata(framed_spectral_kurtosis);
raw_spectral_brightness = mirgetdata(framed_spectral_brightness);
raw_spectral_irregularity = mirgetdata(framed_spectral_irregularity);
raw_spectral_centroid = mirgetdata(framed_spectral_centroid);


%% Labels
window_end=(window/2)*(1:size(raw_rms_energy,2)) + (window/2);
window_m=window_end-window/2;
stamp_end=TimestampData(:,1)+TimestampData(:,2);
stamp= [TimestampData(:,1) stamp_end TimestampData(:,3)];
a=(window_m > stamp(:,1) & window_m < stamp(:,2));
label=10*ones(1,size(a,2));
for i=1:size(a,2)
    if(sum(a(:,i))==1)
        label(i)= TimestampData(a(:,i)==1,3);
    else
        label(i)=666;
    end
end
cor_windows= label~=666;





data = [raw_rms_energy' raw_spectral_roll_off_85' raw_spectral_roll_off_90'...
    raw_MFCCs' raw_zcr' raw_spectral_flatness' raw_spectral_kurtosis'...
    raw_spectral_brightness' raw_spectral_irregularity'...
    raw_spectral_centroid' label'];

[N1,M1] = size(data);

normalized_data = zeros(N1,M1);

for i = 1 : M1-1
    feature_min = min(data(:,i));
    feature_max = max(data(:,i));
    normalized_data(:,i) = (data(:,i) - feature_min)/(feature_max - feature_min);
end
normalized_data(:,M1) = data(:,M1);

data = normalized_data(cor_windows,:);
%%
data = data(~any(isnan(data')),:);

[rank , ~] = relieff(data(:,1:size(data,2)-1),data(:,size(data,2)),20,'method','classification');

%figure('name','pitch')
%plot(raw_pitch)
features = {'rms_energy','roll_off_85','roll_off_90'};
aw = {};
for i = 1:13
    aw{i} = sprintf('MFCC%d',i);
    features{3 + i} = aw{i};
end
features(1,17:22) = {'zcr','spectral_flatness','spectral_kurtosis',...
    'spectral_brightness','spectral_irregularity','spectral_centroid'};
desceding_order_features = features(rank);

figure('name','rms_energy')
plot(raw_rms_energy)
figure('name','roll_off_85')
plot(raw_spectral_roll_off_85)
figure('name','roll_off_90')
plot(raw_spectral_roll_off_90)
figure('name','MFCCs(1:6)')
plot(raw_MFCCs(1:6,:)')
legend(aw{1:6});
figure('name','MFCCs(7:13)')
plot(raw_MFCCs(7:13,:)')
legend(aw{7:13});
figure('name','zcr')
plot(raw_zcr)
figure('name','spectral_flatness')
plot(raw_spectral_flatness)
figure('name','spectral_kurtosis')
plot(raw_spectral_kurtosis)
figure('name','spectral_brightness')
plot(raw_spectral_brightness)
figure('name','spectral_irregularity')
plot(raw_spectral_irregularity)
figure('name','spectral_centroid')
plot(raw_spectral_centroid)


%inharmonicity1 = mirinharmonicity(bagpipe)
%f = mirpitch(bagpipe,'Mono');
% bagpipeinharmonicity = mirinharmonicity(bagpipeSpec,'f0',bagpipeSpecPitch)

%% Part 2 - Classification

[N2,M2] = size(data);
t_opt = [50 0 0.01 0.999995 1.000005];
d_opt = [0 0 0 0];
sbcOptions = [1.25 0.5 0.15 0];

for i = 1 : M2
    feature_min(i) = min(data(:,i));
    feature_max(i) = max(data(:,i));
end

tic;
[rank,w] = relieff(data(:,1:M2-1),data(:,M2),20);
fprintf('Time elapsed for Relieff on feature selection 20 was %.3f sec\n',toc);
xBounds = [feature_min(:)';feature_max(:)'];
% fprintf('Weights: W3=%.7f W6=%.7f W9=%.7f\n', w(rank(NF)));
%% Define sets of model parameter values to evaluate
% NF = [5 10 15 20];
NF = [5 10 15 20];
% NR = [4 8 12 16 20];
NR = [3 6 12 18];


% Preallocations Required for Parallel Processing
allT_Errors = zeros(t_opt(1),5,length(NF)*length(NR));
allC_Errors = zeros(t_opt(1),5,length(NF)*length(NR));
mean_tError = zeros(t_opt(1),length(NF)*length(NR));
mean_cError = zeros(t_opt(1),length(NF)*length(NR));
min_cError = zeros(length(NF), length(NR));
mean_min_cError = zeros(length(NF), length(NR));
TSK_Model_HD(1:5) = newfis('sugeno');
%% Find Radii for each NR
% radius = zeros(length(NF),length(NR));
%% Run This Section
% for i=1:length(NF)
%     features = rank(1:NF(i));
%     for j=1:length(NR)
%         parameter_set_iterator = (i - 1) * length(NR) + j
%         for radii = 1:3
%             increase = radii * 0.001;
%             radiusInput = min(increase+radius(i,j),1);
%             [Centers,~] = subclust(waveform(:,[features M2]),radiusInput,xBounds(:,[features M2]),sbcOptions);
%             NR_new = size(Centers,1)
%             if NR_new == NR(j)
%                 NR_Achieved(i,j) = NR_new;
%                 break;
%             elseif NR_new < NR(j)
%                 radiusInput = radiusInput - increase;
%                 break;
%             end
%         end
%         radius(i,j) = radiusInput;
%     end
% end
%% 
Optimal_radius = [0.76 0.53 0.315 0.271;...
    1 0.73 0.46495 0.39235;...
    1 0.842 0.61965 0.50535;...
    1 0.902 0.718 0.636];
% for i=1:length(NF)
%     features = rank(1:NF(i));
%     for j=1:length(NR)
%         [Cent,~] = subclust(waveform(:,[features M2]),radius(i,j),xBounds(:,[features M2]),sbcOptions);
%         NR_Achieved(i,j) = size(Cent,1);
%     end
% end
for k = 1:30000
    idx = randperm(N2);
    % Shuffle data
    audio_data = data(idx,1:M2);
    % Create Cross Validation Partitions
    CVO = cvpartition(audio_data(:,M2),'KFold',5);
    for l=1:2
        for m = 1:5
            partition_indexes(m,l) = mean(audio_data(CVO.test(m),M2)==l-1);
        end
    end
    % Make Sure all Cross Validation Sets Have Equal Output
    % Frequency
    if any( abs(partition_indexes - 1/2) > 0.005)
        continue;
    else
        break;
    end
end
fprintf('\n%d tries to create CV sets of equal output frequency\n', i);

%% for each parameter set do
for i=1:length(NF)
    features = rank(1:NF(i));
    for j=1:length(NR)
        parameter_set_iterator = (i - 1) * length(NR) + j;
        
        
        tic;
        % Cross Validation
        % for each resampling iteration do
        TSK_model = genfis2(audio_data(:,features),audio_data(:,M2),Optimal_radius(i,j) ,xBounds(:,[features M2]) ,sbcOptions);
        for k=1:length(TSK_model.rule)
            TSK_model.output.mf(k).type = 'constant';
            TSK_model.output.mf(k).params = 1;
        end
        for k = 1:CVO.NumTestSets
            TSK_Model_HD(k) = TSK_model;
        end
        parfor k = 1:CVO.NumTestSets
            strPrint = sprintf('Cross Validation Progress Percentage...%d',(k-1)/CVO.NumTestSets*100);
            fprintf(strPrint);
            % [Optional] Pre-process the data
            
            % Fit the model on the remaidner
            [TSK_Model_HD(k),t_error,~,TSK_Model_CLSS,c_error] = ...
                anfis(audio_data(CVO.training(k),[features M2]), TSK_Model_HD(k), t_opt, d_opt, audio_data(CVO.test(k),[features M2]));
            % Predict the hold-out Samples(anfis does that on "test" Set)
            allT_Errors(:,k,parameter_set_iterator) = t_error;
            allC_Errors(:,k,parameter_set_iterator) = c_error;
            L = length(strPrint);
            fprintf(repmat('\b', 1, L));
        end
        strPrint = sprintf('Cross Validation Progress Percentage...%d',5/CVO.NumTestSets*100);
        fprintf(strPrint);
        fprintf('\nCompleted Cross Val for Parameter Set %d (NF=%d,NR=%d) in %.2f sec\n',parameter_set_iterator,NF(i),NR(j),toc);
        if isnan(any(allC_Errors(:,:,parameter_set_iterator))) | isnan(any(allT_Errors(:,:,parameter_set_iterator)))
            warning('NaN error values have occured');
        end
        % Calculate the average performance across the hold-out predictions
        min_cError(i,j) = min(min(allC_Errors(:,:,parameter_set_iterator)));
        mean_min_cError(i,j) = mean(min(allC_Errors(:,:,parameter_set_iterator)));
    end
end
%% Determine the optimal parameter set
minError = 10;
for i=1:length(NF)
    for j=1:length(NR)
        if mean_min_cError(i,j) < minError
            minError = mean_min_cError(i,j);
            optNF = NF(i);
            optNR = NR(j);
            optRadius = Optimal_radius(i,j);
        end
    end
end
features = [rank(1:optNF) M2];
%% Fit the final model to all training data using the optimal parameter set
[N2,M2] = size(data);
for i = 1 : optNF+1
    feature_min(i) = min(data(:,features(i)));
    feature_max(i) = max(data(:,features(i)));
end
%%
clsNum = feature_max(optNF+1) + 1;
partition_indexes = 0.5 * ones(3,clsNum);
equal_output_partition_error = 0.005;
decreaseFeat = floor(optNF / 2);
flag = true;
while flag
    for i = 1:30000
        idx = randperm(N2);
        D_trn = data(idx(1:ceil(0.6*N2)),features);
        flag = false;
        for j = 1 : length(features) - decreaseFeat
            if any(D_trn(:,j) == feature_min(j)) && any(D_trn(:,j) == feature_max(j))
                continue;
            else
                flag = true;
                break;
            end
        end
        if ~flag
            D_val = data(idx(ceil(0.6*N2)+1:ceil(0.8*N2)),features);
            D_chk = data(idx(ceil(0.8*N2)+1:N2),features);
            for k=1:clsNum
                partition_indexes(:,k) = [mean(D_trn(:,length(features))==k-1); mean(D_val(:,length(features))==k-1); mean(D_chk(:,length(features))==k-1)];
            end
            if any( abs(partition_indexes - 1/clsNum) > equal_output_partition_error)
                continue
            else
                break;
            end
        end
    end
    decreaseFeat = decreaseFeat + 1;
end
fprintf('\n%d tries for data partition, ignored bound values for %d features\n', i, decreaseFeat-1);
fprintf('Achieved output frequency equality %5.2f %%\n',(1-equal_output_partition_error)*100);
%%
t_opt = [300 0 0.01 0.99995 1.00005];
TSK_Model_Final = genfis2(data(:,features(1:optNF)),data(:,M2), optRadius ,xBounds(:,features) ,sbcOptions);
for k=1:length(TSK_Model_Final.rule)
    TSK_Model_Final.output.mf(k).type = 'constant';
    TSK_Model_Final.output.mf(k).params = 1;
end
[x1,y1] = plotmf(TSK_Model_Final,'input',2,600);
[~, t_error, ~, TSK_Model_Final_NoOVT, c_error] = anfis(D_trn, TSK_Model_Final, t_opt, d_opt, D_val);
[x2,y2] = plotmf(TSK_Model_Final_NoOVT,'input',2,600);
%%
for j=1:2
    plotTitle = sprintf('Preview of Input 2 Membership Function %d Alteration',j);
    figure('NumberTitle','off','name',plotTitle);
    plot(x1(:,j),y1(:,j),x2(:,j),y2(:,j))
    xlabel('Input 2', 'FontSize', 18);
    ylabel('Degree of Membership', 'FontSize', 18);
    legend('Initial Model','Trained Model')
end
%% y = D_chk(:,length(features));
N = size(D_chk,1);
yPredicted = evalfis(D_chk(:,1:optNF), TSK_Model_Final_NoOVT);
yActual = D_chk(:,optNF+1);
y = round(yPredicted);
ErrorMatrix = zeros(clsNum);
x_ir = zeros(1,clsNum);
x_jc = zeros(1,clsNum);
PA = zeros(1,clsNum);
UA = zeros(1,clsNum);
for j=1:length(y)
    ErrorMatrix(y(j)+1,yActual(j)+1) = ErrorMatrix(y(j)+1,yActual(j)+1) + 1;
end
% Model Evaluation Parameters
figure('NumberTitle','off','name','Model Misclassification')
[~,sor]=sort(yActual);
plot(abs(y(sor)-yActual(sor)))
xlabel('Samples', 'FontSize', 18);
ylabel('Misclassification', 'FontSize', 18);
OA = sum(sum(eye(clsNum) .* ErrorMatrix(:,:)))/N;
for j=1:clsNum
    x_ir(j) = sum(ErrorMatrix(j,:));
    x_jc(j) = sum(ErrorMatrix(:,j));
    PA(j) = ErrorMatrix(j,j)/x_jc(j);
    UA(j) = ErrorMatrix(j,j)/x_ir(j);
end
K_hat = ((N ^ 2) * OA - sum(x_ir(:) .* x_jc(:)))/(N ^ 2 - sum(x_ir(:) .* x_jc(:)));
fprintf('=-------------------------------=\nNR = %d\nOA = %.6f\nK_hat = %.6f\n',optNR,OA,K_hat);
fprintf('ErrorMatrix = \n%d\t%d\t%d\n%d\t%d\t%d\n%d\t%d\t%d\n',ErrorMatrix(:,:));
fprintf('x_ir =  %d %d %d \nx_jc = %d %d %d\n',x_jc(:),x_jc(:));
fprintf('PA =  %.4f %.4f %.4f \nUA = %.4f %.4f %.4f\n',PA(:),UA(:));
str = sprintf('Learning Curve on Final Model NF = %d, MR = %d using %d Epochs',optNF,optNR,t_opt(1));
figure('NumberTitle','off','Name',str)
plot(c_error);
hold on
plot(t_error);
legend('Check Error','Training Error')
xlabel('Epochs', 'FontSize', 18);
ylabel('RMSE', 'FontSize', 18);
figure('NumberTitle','off','Name','Model Predictions and Actual Values')
plot(y(sor));
hold on
plot(yActual(sor));
legend('Final Model Predictions','Actual Values')
grid on
xlabel('Samples', 'FontSize', 18);
ylabel('Class', 'FontSize', 18);