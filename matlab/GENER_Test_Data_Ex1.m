% DoA estimation via CV-DNN: Experiment 1 -spatial spectrum estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Shulin Hu
% Date: 5/15/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
tic;
rng(14);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Location to save the data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
alpha_str = ["0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1"];
L = 60;
T = 1000; % number of snapreshots
SNR_dB = 20; % SNR values
d_ref = 0.5;
ULA_N = 16;
array_num = length(alpha);
Test_mum = 1000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for i=1:array_num
    L__ = floor(asind(alpha(i)*sind(L)));
    DOA_set = -L__:1:L__;
    d = d_ref/alpha(i);
    
    R_sam = zeros(ULA_N,ULA_N,3,length(DOA_set));
    
    for j =1:length(DOA_set)
        [sam,angles] = Gener_Sam_Covar_Matrix(DOA_set(j),T,SNR_dB,ULA_N,d);
        R_sam(:,:,:,j)=sam;
    end
%     filename = fullfile('E:\code\array_tansfor\data\EX1',sprintf("EX1_Alpha_%s.h5", alpha_str(i)));
%     h5create(filename,'/sam', size(R_sam));
%     h5write(filename, '/sam', R_sam);
%     h5create(filename,'/angle', size(DOA_set));
%     h5write(filename, '/angle', DOA_set);
    delete R_sam;
end

% h5create(filename,'/sam', size(R_sam));
% h5write(filename, '/sam', R_sam);
% h5create(filename,'/angle', size(Angles));
% h5write(filename, '/angle', Angles);
% [sam,angles] = Gener_Sam_Covar_Matrix(DOA_set_K_2_small,T,SNR_dB,ULA.N,d);
% h5create(filename,'/sam', size(sam));
% h5write(filename, '/sam', sam);
% h5create(filename,'/angle', size(angles));
% h5write(filename, '/angle', angles);

% [sam,angles] = Gener_Sam_Covar_Matrix(DOA_set_K_2_lager,T,SNR_dB,ULA.N,d);
% h5create(filename2,'/sam', size(sam));
% h5write(filename2, '/sam', sam);
% h5create(filename2,'/angle', size(angles));
% h5write(filename2, '/angle', angles);


% [sam,angles] = Gener_Sam_Covar_Matrix(DOA_set_K_3,T,SNR_dB,ULA.N,d);
% h5create(filename4,'/sam', size(sam));
% h5write(filename4, '/sam', sam);
% h5create(filename4,'/angle', size(angles));
% h5write(filename4, '/angle', angles);


function [sam,angles] = Gener_Sam_Covar_Matrix(DOAs,T,SNR_dB,ULA_N,d)
    % The steering/response vector of the ULA, where theta=0.5*sin(deg2rad(x));
    ULA_steer_vec = @(x,N,d) exp(1j*2*pi*d*sin(deg2rad(x))*(0:1:N-1)).'; 
    K  = length(DOAs);
    A_ula =zeros(ULA_N,K);
    for k=1:K 
        A_ula(:,k) = ULA_steer_vec(DOAs(k),ULA_N,d);
    end  
    SOURCE.power = ones(1,K).^2;
    noise_power = min(SOURCE.power)*10^(-SNR_dB/10);
    S = (randn(K,T)+1j*randn(K,T))/sqrt(2); 
    X = A_ula*S;
    Eta = sqrt(noise_power)*(randn(ULA_N,T)+1j*randn(ULA_N,T))/sqrt(2);
    Y = X + Eta;
    Ry_sam = Y*Y'/T;
    %------------------------------测试空间旋转矩阵构造------------------------------------------
%     theta_pr = 30;
%     a_theta = ULA_steer_vec(theta_pr,ULA_N,d);   % 接收信号方向向量
%     Ry_sam = diag(a_theta')*Ry_sam*diag(a_theta')';
    %------------------------------测试空间旋转矩阵构造end---------------------------------------
    %------------------------------正交投影矩阵构造------------------------------------------
%     angle_grids = -25:1:25;
%     A_proj = ULA_steer_vec(angle_grids.',ULA_N,d);
%     A_proj = A_proj*(A_proj'*A_proj)*A_proj';
%     Ry_sam = A_proj*Ry_sam;
    %------------------------------正交投影矩阵构造end---------------------------------------
    sam(:,:,1) = real(Ry_sam); 
    sam(:,:,2) = imag(Ry_sam);
    sam(:,:,3) = angle(Ry_sam);
    angles(:) = DOAs';
end
