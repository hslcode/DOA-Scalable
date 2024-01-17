% The mapping in DL_based DoA estimation mathods and its applications;
% Experiment 2: Verification of  the method of generalization for array geometry;
% This code is used to generate data for Experiment 2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Shulin Hu
% Date: 08/15/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Location to save the data
data_file = '../data/EX2/';
if exist(data_file, 'dir') == 7
else
    mkdir(data_file);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 0.5;
alpha_str = ["0.5"];
theta_max = 60;
ULA_N = 16;
T = 1000; % number of snapreshots
SNR_dB = 20; % SNR values
gamma_base = 0.5;
gamma_bias = gamma_base/alpha;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% For 2D CNN and CV-CNN
theta_hat_max = floor(asind(alpha(1)*sind(theta_max)));
DOA_set = -theta_hat_max:1:theta_hat_max;
R_sam_gamma_base = zeros(ULA_N,ULA_N,3,length(DOA_set));
R_sam_gamma_bias = zeros(ULA_N,ULA_N,3,length(DOA_set));

for j =1:length(DOA_set)
    [sam,angles] = Gener_Sam_Covar_Matrix(DOA_set(j),T,SNR_dB,ULA_N,gamma_base);
    R_sam_gamma_base(:,:,1,j)=real(sam);
    R_sam_gamma_base(:,:,2,j)=imag(sam);
    R_sam_gamma_base(:,:,3,j)=angle(sam);
    clear sam;
    clear angles;

    [sam,angles] = Gener_Sam_Covar_Matrix(DOA_set(j),T,SNR_dB,ULA_N,gamma_bias);
    R_sam_gamma_bias(:,:,1,j)=real(sam);
    R_sam_gamma_bias(:,:,2,j)=imag(sam);
    R_sam_gamma_bias(:,:,3,j)=angle(sam);
    clear sam;
    clear angles;
end

filename = fullfile(data_file,sprintf("EX2_Alpha_%s.h5", alpha_str(1)));
h5create(filename,'/angle', size(DOA_set));
h5write(filename, '/angle', DOA_set);

h5create(filename,'/sam_gamma_base', size(R_sam_gamma_base));
h5write(filename, '/sam_gamma_base', R_sam_gamma_base);

h5create(filename,'/sam_gamma_bias', size(R_sam_gamma_bias));
h5write(filename, '/sam_gamma_bias', R_sam_gamma_bias);

clear R_sam_gamma_base;
clear R_sam_gamma_bias;
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
theta_hat_max = floor(asind(alpha(1)*sind(theta_max)));
DOA_set=-theta_hat_max:1:theta_hat_max;
DOA_ture = DOA_set;
DOA_num = length(DOA_set);
S_est_gamma_base=zeros(DOA_num,grids,2);
S_est_gamma_bias=zeros(DOA_num,grids,2);

for k=1:DOA_num
    Rx = Gener_Sam_Covar_Matrix(DOA_set(k),T,SNR_dB,ULA_N,gamma_base);
    temp=H'*reshape(Rx,ULA_N*ULA_N,1);
    temp=temp/norm(temp);
    S_est_gamma_base(k,:,1)=real(temp);
    S_est_gamma_base(k,:,2)=imag(temp);
    clear Rx;
    clear temp;

    Rx = Gener_Sam_Covar_Matrix(DOA_set(k),T,SNR_dB,ULA_N,gamma_bias);
    temp=H'*reshape(Rx,ULA_N*ULA_N,1);
    temp=temp/norm(temp);
    S_est_gamma_bias(k,:,1)=real(temp);
    S_est_gamma_bias(k,:,2)=imag(temp);
    clear Rx;
    clear temp;
end

filename = fullfile(data_file,sprintf("EX2_Alpha_%s.mat", alpha_str(1)));
save(filename,'DOA_ture','S_est_gamma_base',"S_est_gamma_bias")
clear S_est_gamma_base;
clear S_est_gamma_bias;