% The mapping in DL_based DoA estimation mathods and its applications;
% Plot the results of Experiment 3: Statistical Performance Analysis for Fine-Grained Grid;
%  Plot Ratio of RMSE for DoA estimation results at various α.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Shulin Hu
% Date: 08/15/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load the results
f_reslut = fullfile('../result/EX3','Result_EX3_RMSE.h5');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 0.1:0.1:1;

orange = [0.8500, 0.3250, 0.0980];
gold_yellow = [0.9290, 0.6940, 0.1250];
new_green = [0.4660, 0.6740, 0.1880];

Ave_RSME_ratio_DNN = double(h5read(f_reslut, '/Ave_RSME_ratio_DNN'));
Ave_RSME_ratio_1DCNN = double(h5read(f_reslut, '/Ave_RSME_ratio_1DCNN'));
Ave_RSME_ratio_2DCNN = double(h5read(f_reslut, '/Ave_RSME_ratio_2DCNN'));
Ave_RSME_ratio_CVCNN = double(h5read(f_reslut, '/Ave_RSME_ratio_CVCNN'));

figure
hold on;
plot(alpha,Ave_RSME_ratio_DNN,'o-','Color','c','LineWidth',1);
plot(alpha,Ave_RSME_ratio_1DCNN,'*-','Color','b','LineWidth',1);
plot(alpha,Ave_RSME_ratio_2DCNN,'square-','Color',new_green,'LineWidth',1);
plot(alpha,Ave_RSME_ratio_CVCNN,'pentagram-','Color','r','LineWidth',1);
hold off;
grid on;
xlabel("\alpha",'FontName','Times New Roman',"FontSize",14)

ylabel("RMSE ratio",'FontName','Times New Roman',"FontSize",14);
legend("DNN [11]","1D CNN [12]","2D CNN [13]","CV-CNN [14]" ...
    ,'FontName','Times New Roman',"FontSize",12,"FontWeight","bold");
