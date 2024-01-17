% The mapping in DL_based DoA estimation mathods and its applications;
% Plot the results of Experiment 1: The Validation of the Mapping in DL-based DoA Estimator;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Author: Shulin Hu
% Date: 08/15/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load the results
f_reslut = fullfile('../result/EX1','Result_EX1_Phase_Each_DoA_Alpha_0.5.h5');
Ture_phase_DNN = double(h5read(f_reslut, '/Ture_phase_DNN'));
Est_phase_DNN = double(h5read(f_reslut, '/Est_phase_DNN'));
Ture_phase_1D_CNN = double(h5read(f_reslut, '/Ture_phase_1D_CNN'));
Est_phase_1D_CNN = double(h5read(f_reslut, '/Est_phase_1D_CNN'));
Ture_phase_2D_CNN = double(h5read(f_reslut, '/Ture_phase_2D_CNN'));
Est_phase_2D_CNN = double(h5read(f_reslut, '/Est_phase_2D_CNN'));
Ture_phase_CV_CNN = double(h5read(f_reslut, '/Ture_phase_CV_CNN'));
Est_phase_CV_CNN = double(h5read(f_reslut, '/Est_phase_CV_CNN')); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
doa_range = -25:1:25;
orange = [0.8500, 0.3250, 0.0980];
gold_yellow = [0.9290, 0.6940, 0.1250];
new_green = [0.4660, 0.6740, 0.1880];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot Average RMSE of the true SPD versus the estimated SPD at varying geometry
figure
hold on;
plot(doa_range,Ture_phase_1D_CNN,'rd-','LineWidth',1,"MarkerSize",8);
plot(doa_range(1:end-1),Est_phase_DNN,'c+-','LineWidth',1,"MarkerSize",6);
plot(doa_range,Est_phase_1D_CNN,'mpentagram-','LineWidth',1,"MarkerSize",6);
plot(doa_range,Est_phase_2D_CNN,'bo-','LineWidth',1,"MarkerSize",6);
plot(doa_range,Est_phase_CV_CNN,'gx-','LineWidth',1,"MarkerSize",6);
hold off;
grid on;
xlabel("DoA[\circ]","FontSize",14,'FontName','Times New Roman')
ylabel("SPD [rad]","FontSize",14,'FontName','Times New Roman');
legend("Ture SPD","Estimated SPD (DNN)","Estimated SPD (1D CNN)","Estimated SPD (2D CNN)","Estimated SPD (CV-CNN)", ...
    "FontSize",12,'FontName','Times New Roman');
% plot Average RMSE of the true SPD versus the estimated SPD at varying
% geometry with partial detail drawing
figure
hold on;
plot(doa_range,Ture_phase_1D_CNN,'rd-','LineWidth',1,"MarkerSize",8);
plot(doa_range(1:end-1),Est_phase_DNN,'c+-','LineWidth',1,"MarkerSize",6);
plot(doa_range,Est_phase_1D_CNN,'mpentagram-','LineWidth',1,"MarkerSize",6);
plot(doa_range,Est_phase_2D_CNN,'bo-','LineWidth',1,"MarkerSize",6);
plot(doa_range,Est_phase_CV_CNN,'gx-','LineWidth',1,"MarkerSize",6);
hold off;
grid on;
xlim([-5,5]);
%%
%The true SPD and estimated SPD at α=0.5 for four methods.
data = xlsread("../result/EX1/Average_RMSE_Versus_Alpha.xlsx");
alpha = data(:,1);
DNN = data(:,2);
DCN = data(:,3);
CNN = data(:,4);
CV_CNN = data(:,5);
figure
hold on;
plot(alpha,DNN,'o-','Color','c','LineWidth',1);
plot(alpha,DCN,'*-','Color','b','LineWidth',1);
plot(alpha,CNN,'square-','Color',new_green,'LineWidth',1);
plot(alpha,CV_CNN,'pentagram-','Color','r','LineWidth',1);

hold off;
grid on;
xlabel("\alpha","FontSize",14,'FontName','Times New Roman')
ylabel("Average RMSE[rad]","FontSize",14,'FontName','Times New Roman');
legend("DNN","1D CNN","2D CNN","CV-CNN",'FontSize',12,'FontName','Times New Roman');
