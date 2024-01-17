% The mapping in DL_based DoA estimation mathods and its applications;
% This code is used to generate training data for 1D CNN DoA eatimation method.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Shulin Hu
% Date: 08/15/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Location to save the data
data_file = '../data/Train/';
if exist(data_file, 'dir') == 7
else
    mkdir(data_file);
end
filename = fullfile(data_file,'train_data.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ULA_N=16;
snapshot=256;
gamma_base = 0.5;
C=ULA_N*(ULA_N-1);
SNR = 10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DOA11=[];
DOA22=[];
k1=[2:2:40];
k=repmat(k1,1,10);
D_start=-60;
D_stop=59;
for l=1:length(k)
    DOA1=D_start:1:D_stop-k(l);
    DOA2=D_start+k(l):1:D_stop;
    DOA11=[DOA11,DOA1,DOA2];

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DOA_train=[DOA11];
theta=D_start:1:D_stop;
grids=length(theta);
A=exp(1i*2*pi*gamma_base*(0:ULA_N-1)'*sind(theta));
H=zeros(ULA_N*ULA_N,grids);
for i=1:ULA_N
    fhi=A*diag(exp(-1i*pi*(i-1)*sind(theta)));
    H((i-1)*ULA_N+1:i*ULA_N,:)=fhi;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S_label=zeros(length(DOA_train),grids);
S_est=zeros(length(DOA_train),grids,2);
for i=1:length(DOA_train)
    Rx = Gener_Sam_Covar_Matrix(DOA_train(1,i),snapshot,SNR,ULA_N,gamma_base);
    temp=H'*reshape(Rx,ULA_N*ULA_N,1);
    temp=temp/norm(temp);
    S_est(i,:,1)=real(temp);
    S_est(i,:,2)=imag(temp);
    S_label(i,round(DOA11(i))+61)=1;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot(theta,S_est(i,:,1))
xlim([-60,60])
hold on
plot(theta,(S_label(i,:)'))
grid on
legend('A','true')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save(filename,'DOA_train','S_label','S_est')

