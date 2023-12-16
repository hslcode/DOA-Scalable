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
f_reslut = fullfile('../../result/EX2','EX2_result_1D_CNN_d10.5_d21.0.h5');
oneD_CNN_est_DOA_d1 = double(h5read(f_reslut, '/est_DOA_d1'));
oneD_CNN_est_DOA_d2 = double(h5read(f_reslut, '/est_DOA_d2'));

oneD_CNN_est_phase_d1 = double(h5read(f_reslut, '/est_phase_d1'));
oneD_CNN_est_phase_d2 = double(h5read(f_reslut, '/est_phase_d2'));
f_reslut = fullfile('../../result/EX2','EX2_result_2D_CNN_d10.5_d21.0.h5');
twoD_CNN_est_DOA_d1 = double(h5read(f_reslut, '/est_DOA_d1'));
twoD_CNN_est_DOA_d2 = double(h5read(f_reslut, '/est_DOA_d2'));
twoD_CNN_est_phase_d1 = double(h5read(f_reslut, '/est_phase_d1'));
twoD_CNN_est_phase_d2 = double(h5read(f_reslut, '/est_phase_d2'));
f_reslut = fullfile('../../result/EX2','EX2_result_CV_CNN_d10.5_d21.0.h5');
CV_CNN_est_DOA_d1 = double(h5read(f_reslut, '/est_DOA_d1'));
CV_CNN_est_DOA_d2 = double(h5read(f_reslut, '/est_DOA_d2'));
CV_CNN_est_phase_d1 = double(h5read(f_reslut, '/est_phase_d1'));
CV_CNN_est_phase_d2 = double(h5read(f_reslut, '/est_phase_d2'));

f_reslut = fullfile('../../result/EX2','EX2_result_DNN_d10.5_d21.0.h5');
DNN_est_DOA_d1 = double(h5read(f_reslut, '/est_DOA_d1'));
DNN_est_DOA_d2 = double(h5read(f_reslut, '/est_DOA_d2'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
orange = [0.8500, 0.3250, 0.0980];
gold_yellow = [0.9290, 0.6940, 0.1250];
new_green = [0.4660, 0.6740, 0.1880];

alpha = 0.5;
doa_range = -25:1:25;

oneD_CNN_est_DOA_d2_inv = asind(alpha*sind(oneD_CNN_est_DOA_d2));
twoD_CNN_est_DOA_d2_inv = asind(alpha*sind(twoD_CNN_est_DOA_d2));
CV_CNN_est_DOA_d2_inv = asind(alpha*sind(CV_CNN_est_DOA_d2));
DNN_est_DOA_d2_inv = asind(alpha*sind(DNN_est_DOA_d2));
d1 = 0.5;
d2 = 1; %阵列接收间距
a = doa_range;
b = (asind(d2/d1*sind(a)));
%% 画在不同图上
figure
% plot(doa_range,a,'ko','MarkerSize',7.5);
hold on;
% plot(doa_range,b,'bs','MarkerSize',7.5);
plot(doa_range,oneD_CNN_est_DOA_d1(:,1),'rdiamond','LineWidth',1);
plot(doa_range,oneD_CNN_est_DOA_d2(:,1),'go','LineWidth',1);
plot(doa_range,oneD_CNN_est_DOA_d2_inv,'b+','LineWidth',1);
hold off;
grid on;
legend("参考阵列估计","偏差估计(\alpha=0.5)","泛化估计","FontWeight","bold");
xlabel("真实DOA[degree]")
ylabel("估计值[degree]")
title("1D-CNN")
figure
% plot(doa_range,a,'ko','MarkerSize',7.5);
hold on;
% plot(doa_range,b,'bs','MarkerSize',7.5);
plot(doa_range,twoD_CNN_est_DOA_d1(:,1),'rdiamond','LineWidth',1);
plot(doa_range,twoD_CNN_est_DOA_d2(:,1),'go','LineWidth',1);
plot(doa_range,twoD_CNN_est_DOA_d2_inv,'b+','LineWidth',1);
title("2D-CNN")
hold off;
grid on;
legend("参考阵列估计","偏差估计(\alpha=0.5)","泛化估计","FontWeight","bold");
xlabel("真实DOA[degree]")
ylabel("估计值[degree]")
figure
% plot(doa_range,a,'ko','MarkerSize',7.5);
hold on;
% plot(doa_range,b,'bs','MarkerSize',7.5);
plot(doa_range,CV_CNN_est_DOA_d1(:,1),'rdiamond','LineWidth',1);
plot(doa_range,CV_CNN_est_DOA_d2(:,1),'go','LineWidth',1);
plot(doa_range,CV_CNN_est_DOA_d2_inv,'b+','LineWidth',1);
hold off;
grid on;
legend("参考阵列估计","偏差估计(\alpha=0.5)","泛化估计","FontWeight","bold");
xlabel("真实DOA[degree]")
ylabel("估计值[degree]")
title("CV-CNN")

figure
hold on;
plot(doa_range(1:(end-1)),DNN_est_DOA_d1,'rdiamond','LineWidth',1);
plot(doa_range(1:(end-1)),DNN_est_DOA_d2,'go','LineWidth',1);
plot(doa_range(1:(end-1)),DNN_est_DOA_d2_inv,'b+','LineWidth',1);
hold off;
grid on;
legend("参考阵列估计","偏差估计(\alpha=0.5)","泛化估计","FontWeight","bold");
xlabel("真实DOA[degree]")
ylabel("估计值[degree]")
title("DNN")

%% 画在一张
clear all;
close all;
clc;
doa_range = -25:1:25;
gamma1 = 0.5;
alpha = 0.5;
gamma2 = gamma1/alpha;

orange = [0.8500, 0.3250, 0.0980];
gold_yellow = [0.9290, 0.6940, 0.1250];
new_green = [0.4660, 0.6740, 0.1880];
f_reslut = fullfile('../../result/EX2','EX2_result_1D_CNN_d10.5_d21.0.h5');
DOA_ture = double(h5read(f_reslut, '/est_DOA_d1'));
oneD_CNN_est_DOA_d2 = double(h5read(f_reslut, '/est_DOA_d2'));
oneD_CNN_est_DOA_d2_inv = asind(alpha*sind(oneD_CNN_est_DOA_d2));

f_reslut = fullfile('../../result/EX2','EX2_result_2D_CNN_d10.5_d21.0.h5');

twoD_CNN_est_DOA_d2 = double(h5read(f_reslut, '/est_DOA_d2'));
twoD_CNN_est_DOA_d2_inv = asind(alpha*sind(twoD_CNN_est_DOA_d2));

f_reslut = fullfile('../../result/EX2','EX2_result_CV_CNN_d10.5_d21.0.h5');

CV_CNN_est_DOA_d2 = double(h5read(f_reslut, '/est_DOA_d2'));
CV_CNN_est_DOA_d2_inv = asind(alpha*sind(CV_CNN_est_DOA_d2));
figure
hold on;
plot(doa_range,DOA_ture,'rdiamond','LineWidth',1,'MarkerSize',6);
plot(doa_range,oneD_CNN_est_DOA_d2,'r.','LineWidth',1,'MarkerSize',7.5);
plot(doa_range,twoD_CNN_est_DOA_d2,'mx','LineWidth',1,'MarkerSize',7.5);
plot(doa_range,CV_CNN_est_DOA_d2,'pentagram','Color',new_green,'LineWidth',1,'MarkerSize',7.5);

% plot(doa_range,oneD_CNN_est_DOA_d2_inv,'g+','LineWidth',1,'MarkerSize',7.5);
plot(doa_range,twoD_CNN_est_DOA_d2_inv,'b.','LineWidth',1,'MarkerSize',7.5);
% plot(doa_range,CV_CNN_est_DOA_d2_inv,'m*','LineWidth',1,'MarkerSize',7.5);

hold off;
grid on;
% legend("d=0.5","d=1","est-DOA-d_1","est-DOA-d_2");
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
