% The mapping in DL_based DoA estimation mathods and its applications;
% Plot the results of Experiment 2: DoA Estimation Results for Varying Geometry;
%  Plot DoA Estimation Results for Varying Geometry (alpha=0.5).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Shulin Hu
% Date: 08/15/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load the results
theta_max = 60;
alpha = 0.5;
theta_hat_max = floor(asind(alpha*sind(theta_max)));
Ture_DoAs = -theta_hat_max:1:theta_hat_max;
f_reslut = fullfile('../result/EX2','Result_EX2_Geometry_Generalization_Alpha_0.5.h5');

DOA_base_DNN = double(h5read(f_reslut, '/DOA_base_DNN'));
DOA_bias_DNN = double(h5read(f_reslut, '/DOA_bias_DNN'));
DOA_hat_DNN = double(h5read(f_reslut, '/DOA_hat_DNN'));

DOA_base_1DCNN = double(h5read(f_reslut, '/DOA_base_1DCNN'));
DOA_bias_1DCNN = double(h5read(f_reslut, '/DOA_bias_1DCNN'));
DOA_hat_1DCNN = double(h5read(f_reslut, '/DOA_hat_1DCNN'));

DOA_base_2DCNN = double(h5read(f_reslut, '/DOA_base_2DCNN'));
DOA_bias_2DCNN = double(h5read(f_reslut, '/DOA_bias_2DCNN'));
DOA_hat_2DCNN = double(h5read(f_reslut, '/DOA_hat_2DCNN'));


DOA_base_CV_CNN = double(h5read(f_reslut, '/DOA_base_CV_CNN'));
DOA_bias_CV_CNN = double(h5read(f_reslut, '/DOA_bias_CV_CNN'));
DOA_hat_CV_CNN = double(h5read(f_reslut, '/DOA_hat_CV_CNN'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
orange = [0.8500, 0.3250, 0.0980];
gold_yellow = [0.9290, 0.6940, 0.1250];
new_green = [0.4660, 0.6740, 0.1880];
figure(1);
hold on;
plot(Ture_DoAs(1:end-1),DOA_base_DNN,'bdiamond','LineWidth',1);
plot(Ture_DoAs(1:end-1),DOA_bias_DNN,'go','LineWidth',1);
plot(Ture_DoAs(1:end-1),DOA_hat_DNN,'r+','LineWidth',1);
grid on;
legend("Base Estimate with \gamma_{base}","Bias Estimate with \gamma_{bias}","generalization estimate" ...
    ,'FontName','Times New Roman',"FontSize",12,"FontWeight","bold");
xlabel("Ture DoA[\circ]")
ylabel("Estimated DoA[\circ]")

figure(2);
hold on;
plot(Ture_DoAs,DOA_base_1DCNN,'bdiamond','LineWidth',1);
plot(Ture_DoAs,DOA_bias_1DCNN,'go','LineWidth',1);
plot(Ture_DoAs,DOA_hat_1DCNN,'r+','LineWidth',1);
grid on;
legend("Base Estimate with \gamma_{base}","Bias Estimate with \gamma_{bias}","generalization estimate" ...
    ,'FontName','Times New Roman',"FontSize",12,"FontWeight","bold");
xlabel("Ture DoA[\circ]")
ylabel("Estimated DoA[\circ]")

figure(3);
hold on;
plot(Ture_DoAs,DOA_base_2DCNN,'bdiamond','LineWidth',1);
plot(Ture_DoAs,DOA_bias_2DCNN,'go','LineWidth',1);
plot(Ture_DoAs,DOA_hat_2DCNN,'r+','LineWidth',1);
grid on;
legend("Base Estimate with \gamma_{base}","Bias Estimate with \gamma_{bias}","generalization estimate" ...
    ,'FontName','Times New Roman',"FontSize",12,"FontWeight","bold");
xlabel("Ture DoA[\circ]")
ylabel("Estimated DoA[\circ]")

figure(4);
hold on;
plot(Ture_DoAs,DOA_base_CV_CNN,'bdiamond','LineWidth',1);
plot(Ture_DoAs,DOA_bias_CV_CNN,'go','LineWidth',1);
plot(Ture_DoAs,DOA_hat_CV_CNN,'r+','LineWidth',1);
grid on;
legend("Base Estimate with \gamma_{base}","Bias Estimate with \gamma_{bias}","generalization estimate" ...
    ,'FontName','Times New Roman',"FontSize",12,"FontWeight","bold");
xlabel("Ture DoA[\circ]")
ylabel("Estimated DoA[\circ]")
