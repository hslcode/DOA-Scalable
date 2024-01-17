% The mapping in DL_based DoA estimation mathods and its applications;
% Experiment 3: Verification of the method of equivalent finer grid;
% This code is used to generate data for Experiment 3.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Shulin Hu
% Date: 08/15/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Location to save the data
data_file = '../data/EX3/';
if exist(data_file, 'dir') == 7
else
    mkdir(data_file);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
alpha_str = ["0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1"];
theta_max = 60;
T = 1000; % number of snapreshots
Test_mum = 1000;
SNR_dB = 20; % SNR values
gamma_base = 0.5;
ULA_N = 16;
geometry_num = length(alpha);
rng(123);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%For 1D CNN usded to generate the overcomplete matrix
theta=-theta_max:1:theta_max-1;
grids=length(theta);
A=exp(1i*2*pi*gamma_base*(0:ULA_N-1)'*sind(theta));
H=zeros(ULA_N*ULA_N,grids);
for i=1:ULA_N
    fhi=A*diag(exp(-1i*pi*(i-1)*sind(theta)));
    H((i-1)*ULA_N+1:i*ULA_N,:)=fhi;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%For 1D CNN,2D CNN, and CV CNN;
for i=1:geometry_num
    filename = fullfile(data_file,sprintf("EX3_Alpha_%s.h5", alpha_str(i)));
    theta_hat_max = floor(asind(alpha(i)*sind(theta_max)));
    DOA_set = -theta_hat_max:1:theta_hat_max-1;
    DOA_num = length(DOA_set);
    gamma_bias = gamma_base/alpha(i);

    DoA_shift = rand(1,DOA_num).*0.5;       %the off-grid value
    DoA_shift = round(DoA_shift,2);
    DOA_set = DOA_set+DoA_shift;            %the off-grid DoAs
    h5create(filename,'/angle', size(DOA_set));
    h5write(filename, '/angle', DOA_set);

    R_sam_gamma_base = zeros(ULA_N,ULA_N,3,length(DOA_set),Test_mum);
    R_sam_gamma_bias = zeros(ULA_N,ULA_N,3,length(DOA_set),Test_mum);
    S_est_gamma_base=zeros(2,grids,DOA_num,Test_mum);
    S_est_gamma_bias=zeros(2,grids,DOA_num,Test_mum);
    for k = 1:DOA_num
        for j =1:Test_mum
            
            % The data from gamma_base geometry for 2D CNN and CV CNN;
            [sam,~] = Gener_Sam_Covar_Matrix(DOA_set(k),T,SNR_dB,ULA_N,gamma_base);
            R_sam_gamma_base(:,:,1,k,j) = real(sam);
            R_sam_gamma_base(:,:,2,k,j) = imag(sam);
            R_sam_gamma_base(:,:,3,k,j) = angle(sam);
            clear sam;

            % The data from gamma_bias geometry for 2D CNN and CV CNN;
            [sam,~] = Gener_Sam_Covar_Matrix(DOA_set(k),T,SNR_dB,ULA_N,gamma_bias);
            R_sam_gamma_bias(:,:,1,k,j) = real(sam);
            R_sam_gamma_bias(:,:,2,k,j) = imag(sam);
            R_sam_gamma_bias(:,:,3,k,j) = angle(sam);
            clear sam;
            

            % The data from gamma_base geometry for 1D CNN;
            Rx = Gener_Sam_Covar_Matrix(DOA_set(k),T,SNR_dB,ULA_N,gamma_base);
            temp=H'*reshape(Rx,ULA_N*ULA_N,1);
            temp=temp/norm(temp);
            S_est_gamma_base(1,:,k,j)=real(temp);
            S_est_gamma_base(2,:,k,j)=imag(temp);
            clear Rx;
            clear temp;
            
             % The data from gamma_bias geometry for 1D CNN;
            Rx = Gener_Sam_Covar_Matrix(DOA_set(k),T,SNR_dB,ULA_N,gamma_bias);
            temp=H'*reshape(Rx,ULA_N*ULA_N,1);
            temp=temp/norm(temp);
            S_est_gamma_bias(1,:,k,j)=real(temp);
            S_est_gamma_bias(2,:,k,j)=imag(temp);
            clear Rx;
            clear temp;
            
        end
    end
    % save the data of geometry corresponding to alpha(i)
    h5create(filename,'/sam_gamma_base', size(R_sam_gamma_base));
    h5write(filename, '/sam_gamma_base', R_sam_gamma_base);
    h5create(filename,'/sam_gamma_bias', size(R_sam_gamma_bias));
    h5write(filename, '/sam_gamma_bias', R_sam_gamma_bias);
    h5create(filename,'/S_est_gamma_base', size(S_est_gamma_base));
    h5write(filename, '/S_est_gamma_base', S_est_gamma_base);
    h5create(filename,'/S_est_gamma_bias', size(S_est_gamma_bias));
    h5write(filename, '/S_est_gamma_bias', S_est_gamma_bias);
    clear R_sam_gamma_base;
    clear R_sam_gamma_bias;
    clear S_est_gamma_base;
    clear S_est_gamma_bias;
    clear DOA_set;
    disp(i);
end


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%The following code is optional and generates data for 1D CNN and others separately.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% For 1D CNN
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% theta=-theta_max:1:theta_max-1;
% grids=length(theta);
% A=exp(1i*2*pi*gamma_base*(0:ULA_N-1)'*sind(theta));
% H=zeros(ULA_N*ULA_N,grids);
% for i=1:ULA_N
%     fhi=A*diag(exp(-1i*pi*(i-1)*sind(theta)));
%     H((i-1)*ULA_N+1:i*ULA_N,:)=fhi;
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% for i=1:geometry_num
%     theta_hat_max = floor(asind(alpha(i)*sind(theta_max)));
% 
%     DOA_set=-theta_hat_max:1:theta_hat_max-1;
%     DOA_num = length(DOA_set);
%     DoA_shift = rand(1,DOA_num).*0.5;
%     DoA_shift = round(DoA_shift,2);
%     DOA_set = DOA_set+DoA_shift;
%     DOA_ture = DOA_set;
% 
%     gamma_bias = gamma_base/alpha(i);
% 
% 
%     S_est_gamma_base=zeros(Test_mum,DOA_num,grids,2);
%     S_est_gamma_bias=zeros(Test_mum,DOA_num,grids,2);
% 
%     for j = 1:DOA_num
%         for k=1:Test_mum
%             Rx = Gener_Sam_Covar_Matrix(DOA_set(j),T,SNR_dB,ULA_N,gamma_base);
%             temp=H'*reshape(Rx,ULA_N*ULA_N,1);
%             temp=temp/norm(temp);
%             S_est_gamma_base(k,j,:,1)=real(temp);
%             S_est_gamma_base(k,j,:,2)=imag(temp);
%             clear Rx;
% 
%             Rx = Gener_Sam_Covar_Matrix(DOA_set(j),T,SNR_dB,ULA_N,gamma_bias);
%             temp=H'*reshape(Rx,ULA_N*ULA_N,1);
%             temp=temp/norm(temp);
%             S_est_gamma_bias(k,j,:,1)=real(temp);
%             S_est_gamma_bias(k,j,:,2)=imag(temp);
%             clear Rx;
%         end
%     end
%     disp(i);
%     filename = fullfile(data_file,sprintf("EX3_Alpha_%s.mat", alpha_str(i)));
%     save(filename,'DOA_ture','S_est_gamma_base','S_est_gamma_bias')
%     clear S_est_gamma_base;
%     clear S_est_gamma_bias;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                  
% for i=1:geometry_num
%     theta_hat_max = floor(asind(alpha(i)*sind(theta_max)));
% 
%     DOA_set=-theta_hat_max:1:theta_hat_max-1;
%     DOA_num = length(DOA_set);
% 
%     % DOA_set = DOA_set+rand(1,DOA_num);
%     % DOA_ture = DOA_set;
% 
%     gamma_bias = gamma_base/alpha(i);
% 
% 
%     S_est_gamma_base=zeros(Test_mum,DOA_num,grids,2);
%     S_est_gamma_bias=zeros(Test_mum,DOA_num,grids,2);
%     DOA_ture =zeros(Test_mum,DOA_num);
% for k=1:Test_mum
%     theta_hat_max = floor(asind(alpha(i)*sind(theta_max)));
%     DOA_set=-theta_hat_max:1:theta_hat_max-1;
%     DOA_set = DOA_set+rand(1,DOA_num);
%     for j = 1:DOA_num
% 
%             Rx = Gener_Sam_Covar_Matrix(DOA_set(j),T,SNR_dB,ULA_N,gamma_base);
%             temp=H'*reshape(Rx,ULA_N*ULA_N,1);
%             temp=temp/norm(temp);
%             S_est_gamma_base(k,j,:,1)=real(temp);
%             S_est_gamma_base(k,j,:,2)=imag(temp);
%             clear Rx;
% 
%             Rx = Gener_Sam_Covar_Matrix(DOA_set(j),T,SNR_dB,ULA_N,gamma_bias);
%             temp=H'*reshape(Rx,ULA_N*ULA_N,1);
%             temp=temp/norm(temp);
%             S_est_gamma_bias(k,j,:,1)=real(temp);
%             S_est_gamma_bias(k,j,:,2)=imag(temp);
%             clear Rx;
% 
%             DOA_ture(k,j)=DOA_set(j);
%         end
%     end
%     disp(i);
%     filename = fullfile(data_file,sprintf("EX3_Alpha_%s.mat", alpha_str(i)));
%     save(filename,'DOA_ture','S_est_gamma_base','S_est_gamma_bias')
%     clear S_est_gamma_base;
%     clear S_est_gamma_bias;
% end