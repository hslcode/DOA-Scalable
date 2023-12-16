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
f_reslut = fullfile('../../result/EX1','EX1_result_1D_CNN_d10.5_d21.0.h5');
oneD_CNN_est_DOA_d1 = double(h5read(f_reslut, '/est_DOA_d1'));
oneD_CNN_est_DOA_d2 = double(h5read(f_reslut, '/est_DOA_d2'));
oneD_CNN_est_phase_d1 = double(h5read(f_reslut, '/est_phase_d1'));
oneD_CNN_est_phase_d2 = double(h5read(f_reslut, '/est_phase_d2'));
f_reslut = fullfile('../../result/EX1','EX1_result_2D_CNN_d10.5_d21.0.h5');
twoD_CNN_est_DOA_d1 = double(h5read(f_reslut, '/est_DOA_d1'));
twoD_CNN_est_DOA_d2 = double(h5read(f_reslut, '/est_DOA_d2'));
twoD_CNN_est_phase_d1 = double(h5read(f_reslut, '/est_phase_d1'));
twoD_CNN_est_phase_d2 = double(h5read(f_reslut, '/est_phase_d2'));
f_reslut = fullfile('../../result/EX1','EX1_result_CV_CNN_d10.5_d21.0.h5');
CV_CNN_est_DOA_d1 = double(h5read(f_reslut, '/est_DOA_d1'));
CV_CNN_est_DOA_d2 = double(h5read(f_reslut, '/est_DOA_d2'));
CV_CNN_est_phase_d1 = double(h5read(f_reslut, '/est_phase_d1'));
CV_CNN_est_phase_d2 = double(h5read(f_reslut, '/est_phase_d2'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
orange = [0.8500, 0.3250, 0.0980];
gold_yellow = [0.9290, 0.6940, 0.1250];
new_green = [0.4660, 0.6740, 0.1880];

doa_range = -25:1:25;
d1 = 0.5;
d2 = 1; %阵列接收间距
a = doa_range;
b = (asind(d2/d1*sind(a)));
%% 画在不同图上
figure
plot(doa_range,a,'ko','MarkerSize',7.5);
hold on;
plot(doa_range,b,'bs','MarkerSize',7.5);
plot(doa_range,oneD_CNN_est_DOA_d1(:,1),'g+','LineWidth',1);
plot(doa_range,oneD_CNN_est_DOA_d2(:,1),'r.','LineWidth',1);
hold off;
grid on;
legend("d=0.5","d=1","est-DOA-d_1","est-DOA-d_2");
xlabel("真实DOA[degree]")
ylabel("估计值[degree]")
title("1D-CNN")
figure
plot(doa_range,a,'ko','MarkerSize',7.5);
hold on;
plot(doa_range,b,'bs','MarkerSize',7.5);
plot(doa_range,twoD_CNN_est_DOA_d1(:,1),'g+','LineWidth',1);
plot(doa_range,twoD_CNN_est_DOA_d2(:,1),'r.','LineWidth',1);
title("2D-CNN")
hold off;
grid on;
legend("d=0.5","d=1","est-DOA-d_1","est-DOA-d_2",'LineWidth',1);
xlabel("真实DOA[degree]")
ylabel("估计值[degree]")
figure
plot(doa_range,a,'ko','MarkerSize',7.5);
hold on;
plot(doa_range,b,'bs','MarkerSize',7.5);
plot(doa_range,CV_CNN_est_DOA_d1(:,1),'g+','LineWidth',1);
plot(doa_range,CV_CNN_est_DOA_d2(:,1),'r.','LineWidth',1);
hold off;
grid on;
legend("d=0.5","d=1","est-DOA-d_1","est-DOA-d_2");
xlabel("真实DOA[degree]")
ylabel("估计值[degree]")
title("CV-CNN")


%% 画在一张
figure
plot(doa_range,a,'ko','MarkerSize',7.5);
hold on;
plot(doa_range,b,'bs','MarkerSize',7.5);
plot(doa_range,oneD_CNN_est_DOA_d1(:,1),'g+','LineWidth',1,'MarkerSize',7.5);
plot(doa_range,oneD_CNN_est_DOA_d2(:,1),'r.','LineWidth',1,'MarkerSize',7.5);
plot(doa_range,twoD_CNN_est_DOA_d1(:,1),'cdiamond','LineWidth',1,'MarkerSize',7.5);
plot(doa_range,twoD_CNN_est_DOA_d2(:,1),'m^','LineWidth',1,'MarkerSize',7.5);
plot(doa_range,CV_CNN_est_DOA_d1(:,1),'v','Color',gold_yellow,LineWidth',1,'MarkerSize',7.5);
plot(doa_range,CV_CNN_est_DOA_d2(:,1),'pentagram','Color',new_green,'LineWidth',1,'MarkerSize',7.5);
hold off;
grid on;
legend("d=0.5","d=1","est-DOA-d_1","est-DOA-d_2");
xlabel("真实DOA[degree]")
ylabel("估计值[degree]")
%%

%% 画在一张-相位
figure
% plot(doa_range,a,'ko','MarkerSize',7.5);
hold on;
% plot(doa_range,b,'bs','MarkerSize',7.5);
plot(doa_range,oneD_CNN_est_phase_d1(:,1),'g-','LineWidth',1);
plot(doa_range,oneD_CNN_est_phase_d2(:,1),'r--','LineWidth',1);
plot(doa_range,twoD_CNN_est_phase_d1(:,1),'cx','LineWidth',1);
plot(doa_range,twoD_CNN_est_phase_d2(:,1),'m+','LineWidth',1);
plot(doa_range,CV_CNN_est_phase_d1(:,1),'square','Color',gold_yellow,'LineWidth',1);
plot(doa_range,CV_CNN_est_phase_d2(:,1),'pentagram','Color',new_green,'LineWidth',1);
hold off;
grid on;
legend("d=0.5","d=1","est-DOA-d_1","est-DOA-d_2");
xlabel("真实DOA[degree]")
ylabel("估计值[degree]")
%%
clear all;
close all;
clc;
data = xlsread("../../result/EX1/RMSE.xlsx");
alpha = data(:,1);
CV_CNN = data(:,2);
CNN = data(:,3);
DCN = data(:,4);
DNN = data(:,5);
figure
hold on;
plot(alpha,CV_CNN,'gs-','LineWidth',1);
plot(alpha,CNN,'ro-','LineWidth',1);
plot(alpha,DCN,'bx-','LineWidth',1);
plot(alpha,DNN,'md-','LineWidth',1);
hold off;
grid on;
xlabel("\alpha","FontSize",14)
ylabel("RMSE[rad]");
legend("CV-CNN","1D CNN","2D CNN","DNN");
%%
clear all;
close all;
clc;
doa_range = -25:1:25;
f_reslut = fullfile('../../result/EX1','CV_CNN_phase_alpha_0.5.h5');
phase_ture = double(h5read(f_reslut, '/est_phase_d1'));
phase_CV_CNN = double(h5read(f_reslut, '/est_phase_d2'));
f_reslut = fullfile('../../result/EX1','CNN_phase_alpha_0.5.h5');
phase_CNN = double(h5read(f_reslut, '/est_phase_d2'));
f_reslut = fullfile('../../result/EX1','DCN_phase_alpha_0.5.h5');
phase_DCN = double(h5read(f_reslut, '/est_phase_d2'));
f_reslut = fullfile('../../result/EX1','DNN_phase_alpha_0.5.h5');
phase_DNN = double(h5read(f_reslut, '/est_phase_d2'));
figure
hold on;
plot(doa_range,phase_ture,'rd-','LineWidth',1,"MarkerSize",8);
plot(doa_range,phase_CV_CNN,'gx-','LineWidth',1);
plot(doa_range,phase_CNN,'bo-','LineWidth',1);
plot(doa_range,phase_DCN,'mpentagram-','LineWidth',1);
plot(doa_range(1:end-1),phase_DNN,'c+-','LineWidth',1);
hold off;
grid on;
xlabel("DOA[\circ]","FontSize",12)
ylabel("pahse(rad)");
legend("Phase reference","CV-CNN","1D CNN","2D CNN","DNN","FontSize",12);