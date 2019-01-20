load('results_fuzzy.mat');
% According to the standard of IEEE Transactions and Journals: 
% Times New Roman is the suggested font in labels. 
% For a singlepart figure, labels should be in 8 to 10 points,
% whereas for a multipart figure, labels should be in 8 points.
% Width: column width: 8.8 cm; page width: 18.1 cm.
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
set(0,'defaultLineLineWidth',1*k_scaling);
set(0,'defaultAxesLineWidth',0.25*k_scaling);
set(0,'defaultAxesGridLineStyle',':');
set(0,'defaultAxesYGrid','on');
set(0,'defaultAxesXGrid','on');
set(0,'defaultAxesFontName','Times New Roman');
set(0,'defaultAxesFontSize',8*k_scaling);
set(0,'defaultTextFontName','Times New Roman');
set(0,'defaultTextFontSize',8*k_scaling);
set(0,'defaultLegendFontName','Times New Roman');
set(0,'defaultLegendFontSize',8*k_scaling);
set(0,'defaultAxesUnits','normalized');
set(0,'defaultAxesPosition',[left/width bottom/hight (width-left-right)/width  (hight-bottom-top)/hight]);
set(0,'defaultAxesTickDir','out');
set(0,'defaultFigurePaperPositionMode','auto');

%%
figure(1)
p1=plot(wf,OA_wf*100);
xlim([0.1 1]);
p1(1).Marker='*';
xlabel('Window Frame Length (s)');
ylabel('Accuracy (%)');

figure(2)
p1=plot(Kfolds,OA_Kfolds*100);
xlim([3 21]);
p1(1).Marker='*';
xlabel('K Folds');
ylabel('Accuracy (%)');

figure(3)
p1=plot(NF,OA_NF*100);
xlim([2 20]);
ylim([67.49 100]);
p1(1).Marker='*';
xlabel('Number of Best Features');
ylabel('Accuracy (%)');

figure(4)
p1=plot(NR,OA_NR*100);
xlim([2 20]);
p1(1).Marker='*';
xlabel('Number of Rules');
ylabel('Accuracy (%)');

figure(4)
c=categorical({'REP Tree','IBK', 'MLP','Random Committee','Random Forest'});
oa=[89.36 92.55 90.86 91.70 93.24];
xlabel('Model');
ylabel('Accuracy (%)');
ylim([88 100])
bar(c,oa);