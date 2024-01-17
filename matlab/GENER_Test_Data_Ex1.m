% The mapping in DL_based DoA estimation mathods and its applications;
% Experiment 1: Verification for the mapping;
% This code is used to generate data for Experiment 1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Shulin Hu
% Date: 08/15/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Location to save the data
data_file = '../data/EX1/';
if exist(data_file, 'dir') == 7
else
    mkdir(data_file);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%geometry and source parameters
alpha = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
alpha_str = ["0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1"];
theta_max = 60;
T = 1000; % number of snapreshots
SNR_dB = 20; % SNR values
gamma_base = 0.5;
ULA_N = 16;
geometry_num = length(alpha);
test_num = 1000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% For 2D CNN and CV-CNN
for i=1:geometry_num
    theta_hat_max = floor(asind(alpha(i)*sind(theta_max)));
    DOA_set = -theta_hat_max:1:theta_hat_max;
    gamma_bias = gamma_base/alpha(i);

    R_sam = zeros(ULA_N,ULA_N,3,length(DOA_set),test_num);

    for j =1:length(DOA_set)
        for k=1:test_num
            [sam,angles] = Gener_Sam_Covar_Matrix(DOA_set(j),T,SNR_dB,ULA_N,gamma_bias);
            R_sam(:,:,1,j,k)=real(sam);
            R_sam(:,:,2,j,k)=imag(sam);
            R_sam(:,:,3,j,k)=angle(sam);
        end
    end
    filename = fullfile(data_file,sprintf("EX1_Alpha_%s.h5", alpha_str(i)));
    h5create(filename,'/sam', size(R_sam));
    h5write(filename, '/sam', R_sam);
    h5create(filename,'/angle', size(DOA_set));
    h5write(filename, '/angle', DOA_set);
    clear R_sam;
end

%% For 1D CNN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theta=-theta_max:1:theta_max-1;
grids=length(theta);
A=exp(1i*2*pi*gamma_base*(0:ULA_N-1)'*sind(theta));
H=zeros(ULA_N*ULA_N,grids);
for i=1:ULA_N
    fhi=A*diag(exp(-1i*pi*(i-1)*sind(theta)));
    H((i-1)*ULA_N+1:i*ULA_N,:)=fhi;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                  
for i=1:geometry_num
    theta_hat_max = floor(asind(alpha(i)*sind(theta_max)));
    DOA_set=-theta_hat_max:1:theta_hat_max;
    DOA_ture = DOA_set;
    gamma_bias = gamma_base/alpha(i);
    DOA_num = length(DOA_set);
    S_est=zeros(test_num,DOA_num,grids,2);

    for j = 1:test_num
        for k=1:DOA_num
            Rx = Gener_Sam_Covar_Matrix(DOA_set(k),T,SNR_dB,ULA_N,gamma_bias);
            temp=H'*reshape(Rx,ULA_N*ULA_N,1);
            temp=temp/norm(temp);
            S_est(j,k,:,1)=real(temp);
            S_est(j,k,:,2)=imag(temp);
        end
    end
    disp(i);
    filename = fullfile(data_file,sprintf("EX1_Alpha_%s.mat", alpha_str(i)));
    save(filename,'DOA_ture','S_est')
    clear S_est;
end