% TESTING DATA Generator - Experiment 2 (i.e., RMSE varies with SNR.)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Georgios K. Papageorgiou
%Modified by: Shulin Hu
% Date: 15/05/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
%clc;
tic;
rng(14);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the data
data = xlsread("E:\code\array_tansfor\result\EX3\RMSE.xlsx");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
orange = [0.8500, 0.3250, 0.0980];
gold_yellow = [0.9290, 0.6940, 0.1250];
new_green = [0.4660, 0.6740, 0.1880];

doa_range = -25:1:25;
d1 = 0.5;
d2 = 1; %阵列接收间距
a = doa_range;
b = (asind(d2/d1*sind(a)));

alpha = data(:,1);

DCN_1 = data(:,2);
DCN_2 = data(:,3);
DNN_1 = data(:,4);
DNN_2 = data(:,5);
CV_CNN_1 = data(:,6);
CV_CNN_2 = data(:,7);
CNN_1 = data(:,8);
CNN_2 = data(:,9);

DCN = DCN_2./DCN_1;
DNN = DNN_2./DNN_1;
CV_CNN = CV_CNN_2./CV_CNN_1;
CNN = CNN_2./CNN_1;
figure
hold on;
plot(alpha,CV_CNN,'gs-','LineWidth',1);
plot(alpha,CNN,'ro-','LineWidth',1);
plot(alpha,DCN,'bx-','LineWidth',1);
plot(alpha,DNN,'md-','LineWidth',1);
hold off;
grid on;
xlabel("\alpha","FontSize",14)
ylabel("RMSE比值");
legend("CV-CNN","2D CNN","1D CNN","DNN");
