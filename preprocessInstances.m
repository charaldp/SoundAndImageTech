%% width & hight of the figure
k_scaling = 4;          % scaling factor of the figure
% (You need to plot a figure which has a width of (8.8 * k_scaling)
% in MATLAB, so that when you paste it into your paper, the width will be
% scalled down to 8.8 cm  which can guarantee a preferred clearness.
k_width_hight = 2;      % width:hight ratio of the figure
width = 8.8 * k_scaling;
hight = width / k_width_hight;
%% figure margins
top = 0.5;  % normalized top margin
bottom = 3;	% normalized bottom margin
left = 3.5;	% normalized left margin
right = 1;  % normalized right margin
%% set default figure configurations
set(0,'defaultFigureUnits','centimeters');
set(0,'defaultFigurePosition',[0 0 width hight]);
set(0,'defaultLineLineWidth',k_scaling/4);
set(0,'defaultAxesLineWidth',0.25*k_scaling);
set(0,'defaultAxesGridLineStyle',':');
set(0,'defaultAxesYGrid','on');
set(0,'defaultAxesXGrid','on');
set(0,'defaultAxesFontName','Times New Roman');
set(0,'defaultAxesFontSize',12);
set(0,'defaultTextFontName','Times New Roman');
set(0,'defaultTextFontSize',12);
set(0,'defaultLegendFontName','Times New Roman');
set(0,'defaultLegendFontSize',12);
set(0,'defaultAxesUnits','normalized');
set(0,'defaultAxesPosition',[left/width bottom/hight (width-left-right)/width  (hight-bottom-top)/hight]);
set(0,'defaultAxesTickDir','out');
set(0,'defaultFigurePaperPositionMode','auto');

%% Attribute Names for Weka
close all;
features = {'RMS Energy','Roll Off 85','Roll Off 90'};
for i = 1:13
    features{3 + i} = sprintf('MFCC %d',i);
end
features(17:22) = {'Zero Crossing Rate','Spectral Flatness','Spectral Kurtosis',...
    'Spectral Brightness','Spectral Irregularity','Spectral Centroid'};



removeOutliers = true;
% Outlier are considered the values are out of the standard deviation from
% the mean value of the current feature by 4 - 6 times
outlierMultiplier = 5;

fullData = [];

dataGTZAN = csvread('Output/noATTRSrawDataGTZAN.csv');

dataMusan = csvread('Output/noATTRSrawDataMusan.csv');

csvList = dir('Output/noATTRSrawDataMirex*.csv');

% Dataset plot colors generated using rand()

col = [0.557387833662972 0.277676113578610 0.851704241909172 0.427719595598367 0.149772395260079 0.674817983932854 0.193303818875548 0.715230208959454 0.238729211562486;...
    0.550569082925633 0.751492822015600 0.745906861336381 0.0919877602545426 0.451479522691271 0.674750636263092 0.950836326434131 0.766532794300762 0.904336177144639;...
    0.657382751451916 0.380550311396241 0.801765535326145 0.671013553059228 0.0782190256271231 0.766475896904275 0.242499605325159 0.323701040623197 0.905828900354372];

for feat = 1:22
figure('name',char(features(feat)));
plot(dataGTZAN(:,feat),'x','MarkerSize',2,'Color',col(:,1));
ylabel(sprintf('%s',char(features(feat))));
xlabel('Instance#');
hold on;
mn = min(dataGTZAN(:,feat));
mx = max(dataGTZAN(:,feat));
plot([1 size(dataGTZAN,1) size(dataGTZAN,1) 1 1],[mn mn mx mx mn],'LineWidth',1.5,'Color',col(:,1))
plot(dataMusan(:,feat),'x','MarkerSize',2,'Color',col(:,2));
mn = min(dataMusan(:,feat));
mx = max(dataMusan(:,feat));
plot([1 size(dataMusan,1) size(dataMusan,1) 1 1],[mn mn mx mx mn],'LineWidth',1.5,'Color',col(:,2))
aw2(1:4)={'GTZAN','GTZAN Lims','Musan Corpus','Musan Corpus Lims'};
for j = 1:length(csvList)
    dataMirex = csvread(sprintf('Output/%s',csvList(j).name));
    plot(dataMirex(:,feat),'x','MarkerSize',2,'Color',col(:,2+j));
    mn = min(dataMirex(:,feat));
    mx = max(dataMirex(:,feat));
    plot([1 size(dataMirex,1) size(dataMirex,1) 1 1],[mn mn mx mx mn],'LineWidth',1.5,'Color',col(:,2+j))
    aw2{5 + 2 * (j-1)} = sprintf('Mirex Example: %d',j);
    aw2{6 + 2 * (j-1)} = sprintf('Mirex Example: %d Lims',j);
end
legend(aw2);
end

[N1,M1] = size(dataGTZAN);
% RMS Energy(1) should be normalized separately as concluded by observation
normalize_common = ones(1,23);
normalize_common([1 18 19]) = 0;

GTZANLoss = 0;
for i = 1 : M1-1
    if removeOutliers
        stdDev = sqrt(mean((dataGTZAN(:,i) - mean(dataGTZAN(:,i))).^2));
        remaining = abs(dataGTZAN(:,i) - mean(dataGTZAN(:,i))) < outlierMultiplier * stdDev;
        dataGTZAN = dataGTZAN(remaining,:);
        if(sum(remaining) < N1)
            fprintf('Removed %d outliers on %s on GTZAN\n',N1 - sum(remaining),char(features(i)));
            GTZANLoss = GTZANLoss + N1 - sum(remaining);
        end
    end
    feature_min(i) = min(dataGTZAN(:,i));
    feature_max(i) = max(dataGTZAN(:,i));
    dataGTZAN(:,i) = (dataGTZAN(:,i) - feature_min(i))/(feature_max(i) - feature_min(i));
    N1 = size(dataGTZAN,1);
end
fullData = [fullData;dataGTZAN];

MusanLoss = 0;
[N2,M2] = size(dataMusan);
for i = 1 : M2-1
    if removeOutliers
        stdDev = sqrt(mean((dataMusan(:,i) - mean(dataMusan(:,i))).^2));
        remaining = abs(dataMusan(:,i) - mean(dataMusan(:,i))) < outlierMultiplier * stdDev;
        dataMusan = dataMusan(remaining,:);
        if(sum(remaining) < N2)
            fprintf('Removed %d outliers on %s on Musan Corpus\n',N2 - sum(remaining),char(features(i)));
            MusanLoss = MusanLoss + N2 - sum(remaining);
        end
    end
    if(normalize_common(i))
        dataMusan(:,i) = (dataMusan(:,i) - feature_min(i))/(feature_max(i) - feature_min(i));
    else
        temp_min = min(dataMusan(:,i));
        temp_max = max(dataMusan(:,i));
        dataMusan(:,i) = (dataMusan(:,i) - temp_min)/(temp_max - temp_min);
    end
    N2 = size(dataMusan,1);
end
fullData = [fullData;dataMusan];
     
%% Mirex Dataset Common Normalize and Combine

dataMirex = [];
MirexLoss = 0;
for j = 1:length(csvList)
    dataTemp = csvread(sprintf('Output/%s',csvList(j).name));
    [N3(j),M3] = size(dataTemp);
    for i = 1 : M3-1
        if removeOutliers
            stdDev = sqrt(mean((dataTemp(:,i) - mean(dataTemp(:,i))).^2));
            remaining = abs(dataTemp(:,i) - mean(dataTemp(:,i))) < outlierMultiplier * stdDev;
            dataTemp = dataTemp(remaining,:);
            if(sum(remaining) < N3(j))
                fprintf('Removed %d outliers on %s on Mirex Example %d\n',N3(j) - sum(remaining),char(features(i)),j);
                MirexLoss = MirexLoss + N3(j) - sum(remaining);
            end
        end
        if(normalize_common(i))
            dataTemp(:,i) = (dataTemp(:,i) - feature_min(i))/(feature_max(i) - feature_min(i));
        else
            temp_min = min(dataTemp(:,i));
            temp_max = max(dataTemp(:,i));
            dataTemp(:,i) = (dataTemp(:,i) - temp_min)/(temp_max - temp_min);
        end
        N3(j) = size(dataTemp,1);
    end
    dataMirex = [dataMirex;dataTemp];
end
fullData = [fullData;dataMirex];

if abs(mean(dataGTZAN(:,23)) - 0.5) > 0.1
    warning('Class Inequality on GTZAN dataset!');
    fprintf('Mean Class = %1.2f\n',mean(dataGTZAN(:,23)));
end
fprintf('Instance Loss %.2f %% on GTZAN dataset from outlier removal\n',GTZANLoss/(GTZANLoss + size(dataGTZAN,1))*100);

if abs(mean(dataMusan(:,23)) - 0.5) > 0.1
    warning('Class Inequality on Musan dataset!');
    fprintf('Mean Class = %1.2f\n',mean(dataMusan(:,23)));
end
fprintf('Instance Loss %.2f %% on Musan dataset from outlier removal\n',MusanLoss/(MusanLoss + size(dataMusan,1))*100);

if abs(mean(dataMirex(:,23)) - 0.5) > 0.1
    warning('Class Inequality on Mirex Examples dataset!');
    fprintf('Mean Class = %1.2f\n',mean(dataMirex(:,23)));
end
fprintf('Instance Loss %.2f %% on Mirex dataset from outlier removal\n',MirexLoss/(MirexLoss + size(dataMirex,1))*100);

for feat = 1:22
    figure('name',sprintf('Normalized %s on Full Data',char(features(feat))));
    plot(fullData(:,feat))
    ylabel(sprintf('%s',char(features(feat))));
    xlabel('Instance#');
end
%% Extract Preprocessed Datasets
features = {'RMS_Energy','Roll_Off_85','Roll_Off_90'};
for i = 1:13
    features{3 + i} = sprintf('MFCC_%d',i);
end
features(17:22) = {'Zero_Crossing_Rate','Spectral_Flatness','Spectral_Kurtosis',...
    'Spectral_Brightness','Spectral_Irregularity','Spectral_Centroid'};
attributes = features;
attributes(23) = {'Class'};

writetable(cell2table([attributes;num2cell(dataGTZAN)]),'Output/afterProcGTZAN.csv','writevariablenames',0);
writetable(cell2table(num2cell(dataGTZAN)),'Output/afterProcNoAttrsGTZAN.csv','writevariablenames',0);

writetable(cell2table([attributes;num2cell(dataMusan)]),'Output/afterProcMusan.csv','writevariablenames',0);
writetable(cell2table(num2cell(dataMusan)),'Output/afterProcNoAttrsMusan.csv','writevariablenames',0);

writetable(cell2table([attributes;num2cell(dataMirex)]),'Output/afterProcMirex.csv','writevariablenames',0);
writetable(cell2table(num2cell(dataMirex)),'Output/afterProcNoAttrsMirex.csv','writevariablenames',0);

% writetable(cell2table([attributes;num2cell(fullData)]),'NormalOptFullData.csv','writevariablenames',0);
%% View Ranking of features on each dataset

% tic;
% [Rank1,~] = relieff(dataGTZAN(:,1:M2-1),dataGTZAN(:,M2),2,'method','classification');  
% fprintf('Time elapsed for Relieff on feature ranking was %.3f sec\n',toc);
% decendingFeautures1 = features(Rank1);
% 
% tic;
% [Rank2,~] = relieff(dataMusan(:,1:M2-1),dataMusan(:,M2),1,'method','classification');  
% fprintf('Time elapsed for Relieff on feature ranking was %.3f sec\n',toc);
% decendingFeautures2 = features(Rank2);
% 
% tic;
% [Rank3,~] = relieff(dataMirex(:,1:M2-1),dataMirex(:,M2),1,'method','classification');  
% fprintf('Time elapsed for Relieff on feature ranking was %.3f sec\n',toc);
% decendingFeautures3 = features(Rank3);