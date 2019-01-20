function [accuracy,foldsAccuracy,TSK_Model_HD_Opt,rank] = classificationFuzzy(data,NF,NR,K_folds,clust_type)
% [data] features(columns) must be inputed in descending ranked order!
% NF: Number of features used as input vector for FIS model
% NR: Number of rules in the FIS model
% K_folds: Number of Folds in Cross Validation
% clust_type: 0 for Subtractive Clustering, 1 for Fuzzy C-Means (no Radius parameters needed)
% Function is implemented for binary classification

% K_folds = 5;
% clust_type = 1;
% NF = [5 10 15 20];
% NR = [3 6 12 18];
%% Part 2 - Fuzzy Classification
%data = csvread('dataMixedNoAttr.csv');

[N2,M2] = size(data);
t_opt = [30 0 0.01 0.999995 1.000005];

d_opt = [0 0 0 0];
sbcOptions = [1.25 0.5 0.15 0];
fcmOptions = [2 200 1e-5 0];

for i = 1 : M2
    feature_min(i) = min(data(:,i));
    feature_max(i) = max(data(:,i));
end
clsNum = feature_max(M2) + 1;
tic;
[rank,~] = relieff(data(:,1:M2-1),data(:,M2),4,'method','classification');
fprintf('Time elapsed for Relieff on feature selection was %.3f sec\n',toc);
xBounds = [feature_min(:)';feature_max(:)'];


% Preallocations Required for Parallel Processing
accuracy = zeros(length(NF), length(NR));
TSK_Model_HD(1:K_folds) = newfis('sugeno');
TSK_Model_HD_Opt(1:K_folds,1:length(NF),1:length(NR)) = newfis('sugeno');
%% Optimal Radius for genfis2()
% Radius Required for extracting a desired number of model Rules have been 
% optimized by using the script Radii.m The desired radius is obtained
% from column (NR desired) and row (NF desired). Radius for features 20:22
% has not been found due to the increase of complexity on a large number of
% features.

% On update a radius for achieving the desired number of rules is found for 
% the provided training dataset at lines 93 - 119

% Optimal_radius = [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5;...
% 0.5 0.26 0.1975 0.135 0.0725 0.056875 0.045156 0.04125 0.037344 0.036367 0.033438 0.032858 0.032461 0.031738 0.029531 0.028555 0.027578 0.02709 0.026602 0.025625;...
% 0.5 0.385 0.26 0.1975 0.14672 0.135 0.088125 0.0725 0.064688 0.060781 0.059805 0.058828 0.056875 0.052236 0.051016 0.049063 0.047227 0.04625 0.044297 0.042344;...
% 0.5 0.395 0.272 0.1492 0.12037 0.10741 0.1035 0.099595 0.091782 0.087876 0.08397 0.079087 0.078598 0.076157 0.074204 0.073227 0.072251 0.068345 0.066391 0.065415;...
% 0.5 0.275 0.1525 0.12791 0.11921 0.11082 0.10887 0.10691 0.10301 0.10106 0.099102 0.08836 0.083477 0.082501 0.082013 0.081769 0.081524 0.075665 0.073712 0.073224;...
% 0.5 0.28 0.158 0.13005 0.11806 0.11131 0.11082 0.103 0.099097 0.096716 0.096167 0.09519 0.088354 0.087378 0.083472 0.081518 0.081396 0.080542 0.080048 0.079565;...
% 0.5 0.2225 0.181 0.14975 0.13412 0.12729 0.12071 0.1158 0.11164 0.10773 0.10633 0.10371 0.10223 0.10158 0.099819 0.096889 0.094427 0.090697 0.092955 0.087606;...
% 0.5 0.2275 0.16947 0.165 0.13365 0.13035 0.13023 0.1265 0.1195 0.1156 0.11802 0.10974 0.10638 0.10779 0.10443 0.1024 0.10102 0.10067 0.098569 0.098087;...
% 0.5 0.26375 0.2325 0.17 0.15794 0.15013 0.14212 0.13864 0.1345 0.12712 0.12516 0.1152 0.1193 0.11345 0.11198 0.11089 0.11149 0.10991 0.10894 0.10547;...
% 0.5 0.2375 0.23625 0.205 0.1905 0.18269 0.17878 0.175 0.15925 0.128 0.12776 0.12692 0.12448 0.12399 0.12301 0.11931 0.1152 0.11448 0.11442 0.11345;...
% 0.5 0.27375 0.2425 0.20128 0.19738 0.18768 0.18 0.1505 0.1408 0.138 0.12672 0.1233 0.12284 0.12176 0.12088 0.11991 0.11893 0.11795 0.11747 0.11728;...
% 0.5 0.27875 0.2475 0.23272 0.22881 0.185 0.17899 0.16923 0.1536 0.14648 0.14551 0.14063 0.13672 0.13281 0.13086 0.12891 0.12689 0.12683 0.125 0.12598;...
% 0.5 0.27875 0.2915 0.229 0.21291 0.19728 0.18359 0.17188 0.16016 0.15625 0.15214 0.14824 0.1454 0.13478 0.13922 0.1342 0.13531 0.13086 0.12939 0.1275;...
% 0.5 0.57 0.28387 0.237 0.22328 0.20765 0.18923 0.17751 0.16992 0.16895 0.16406 0.16281 0.16189 0.15625 0.15039 0.14515 0.13867 0.13672 0.13379 0.13293;...
% 0.5 0.45 0.325 0.233 0.21809 0.20247 0.19433 0.18847 0.17969 0.17578 0.17188 0.16406 0.15771 0.15748 0.15552 0.15234 0.14844 0.14609 0.14355 0.14219;...
% 0.5 0.58 0.378 0.3027 0.2402 0.2118 0.19296 0.18906 0.18124 0.17969 0.17188 0.16589 0.16016 0.15723 0.15625 0.15234 0.1493 0.14854 0.14844 0.14604;...
% 0.5 0.335 0.3724 0.26641 0.23516 0.20527 0.1949 0.19099 0.1875 0.17578 0.17383 0.15863 0.15625 0.15576 0.1543 0.15234 0.14805 0.14844 0.1475 0.14609;...
% 0.5 0.59 0.394 0.2546 0.24164 0.21367 0.19464 0.19074 0.18292 0.17578 0.16796 0.16406 0.15625 0.15613 0.1543 0.15117 0.15 0.14945 0.1475 0.14844;...
% 0.5 0.345 0.27743 0.2618 0.21687 0.20644 0.19439 0.19081 0.18562 0.16831 0.16406 0.15956 0.15625 0.15479 0.1543 0.15391 0.15324 0.14844 0.14877 0.14486;...
% 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.14648 0.14551 0.14063 0.13672 0.13281 0.13086 0.12891 0.12689 0.12683 0.125 0.12598;...
% 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.14648 0.14551 0.14063 0.13672 0.13281 0.13086 0.12891 0.12689 0.12683 0.125 0.12598;...
% 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.14648 0.14551 0.14063 0.13672 0.13281 0.13086 0.12891 0.12689 0.12683 0.125 0.12598];

idx = randperm(N2);
% Shuffle data
audio_data = data(idx,:);
% Create Cross Validation Partitions
CVO = cvpartition(audio_data(:,M2),'KFold',K_folds);

%% for each parameter set do
for i=1:length(NF)
    features = rank(1:NF(i));
    for j=1:length(NR)
        parameter_set_iterator = (i - 1) * length(NR) + j;
        % Training epochs are adjusted according to the number of rules for a reasonable execution time
        t_opt(1) = 1000/(NR(j) + 20);
        tic;
        % Cross Validation
        % for each resampling iteration do
        if(clust_type)
            % Initialize a FIS model using FCM
            TSK_model = genfis3(audio_data(:,features),audio_data(:,M2), 'sugeno', NR(j),fcmOptions);
        else
            % Initialize a FIS model using SC
            % Find an opitmal radius for SC on the fly
            fprintf('Optimizing subclust Radius for a desired number of model Rules...\n');
            lim = NF(i)/200;
            NR_Achieved = 1;
            Optimal_radius = 0.5;
            for l = 1:50
                if NR_Achieved < NR(j)
                    k = -1;
                else
                    k = 1;
                end
                difference = k*0.5/(2^l);
                Optimal_radius = max(min(Optimal_radius + difference,1),lim);
                [Centers,~] = subclust(audio_data(1:min(12000,size(audio_data,1)),[features M2]),Optimal_radius,xBounds(:,[features M2]),sbcOptions);
                NR_Achieved = size(Centers,1);
                if(NR_Achieved > 300)
                    lim = lim * 1.6;
                elseif(l == 1 && NR_Achieved < 20)
                    lim = lim * 0.9;
                end
                if(NR_Achieved > 20 && NR_Achieved < 300)
                    lim = Optimal_radius;
                end
                fprintf('Target : %d - Achieved: %d\n',NR(j),NR_Achieved);
                if(NR_Achieved==NR(j))
                    break;
                end
            end
            
            TSK_model = genfis2(audio_data(:,features),audio_data(:,M2),...
                Optimal_radius,xBounds(:,[features M2]) ,sbcOptions);
            if(length(TSK_model.rule)~=NR(j))
                % This mismatch might occur when part of the data is provided
                % during subclustering for complexity reduction
                warning('Failed to achieve the desired number of rules');
                fprintf('Target : %d - Achieved: %d\n',NR(j),length(TSK_model.rule));
            end
            for k=1:length(TSK_model.rule)
                TSK_model.output.mf(k).type = 'constant';
                TSK_model.output.mf(k).params = 1;
            end
        end
        
        ErrorMatrix = zeros(clsNum,clsNum,K_folds);
        parfor k = 1:CVO.NumTestSets
            % Fit the model on the remaidner
            [~,~,~,TSK_Model_HD(k,i,j),~] = ...
                anfis(audio_data(CVO.training(k),[features M2]), TSK_model, t_opt, d_opt, audio_data(CVO.test(k),[features M2]));
        end
        
%         Check Learning Curves
%         figure('name','Learning Curve')
%         plot(1:t_opt(1),t_error,1:t_opt(1),chk_error);
%         for k = 1:CVO.NumTestSets  
%             legend('Training Error','Check Error');
%         end
        for k = 1:CVO.NumTestSets
            % Predict the hold-out Samples(anfis does that on "test" Set)
            N(k) = size(audio_data(CVO.test(k),1),1);
            yPredicted = evalfis(audio_data(CVO.test(k),features), TSK_Model_HD(k));
            y = min(max(round(yPredicted),0),1);
            yActual = audio_data(CVO.test(k),M2);
            for l=1:length(y)
                ErrorMatrix(y(l)+1,yActual(l)+1,k) = ErrorMatrix(y(l)+1,yActual(l)+1,k) + 1;
            end
            % Model Evaluation Metrics

            OA(k) = sum(sum(eye(clsNum) .* ErrorMatrix(:,:,k)))/N(k);
            TSK_Model_HD_Opt(k,i,j) = TSK_Model_HD(k);
        end
        fprintf('Completed Cross Val for Parameter Set %d (NF=%d,NR=%d) in %.2f sec\n',parameter_set_iterator,NF(i),NR(j),toc);
        
        accuracy(i,j) = sum(OA.*N)/sum(N);
        foldsAccuracy(:,parameter_set_iterator) = OA;
    end
end