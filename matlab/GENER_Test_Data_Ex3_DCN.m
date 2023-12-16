%% 产生数据
clc
clear variables
close all
alpha = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
alpha_str = ["0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1"];
array_num = length(alpha);
M=16;
snapshot=1000;
f0=1e6;
fc=1e6;
fs=4*f0;
C=M*(M-1);
D_start=-60;
D_stop=59;
L = 60;
SNR = 10;
test_num = 1000;

theta=D_start:1:D_stop;
grids=length(theta);
A=exp(1i*pi*fc*(0:M-1)'*sind(theta)/f0);
H=zeros(M*M,grids);
for i=1:M
    fhi=A*diag(exp(-1i*pi*(i-1)*sind(theta)));
    H((i-1)*M+1:i*M,:)=fhi;
end
                        


for i=1:array_num
    L__ = floor(asind(alpha(i)*sind(L)));
    DOA=-L__:1:L__-1;
    DOA_num = length(DOA);
    S_est_d1=zeros(DOA_num,test_num,grids,2);
    S_est_d2=zeros(DOA_num,test_num,grids,2);
    DOA_ture = zeros(test_num,DOA_num);
    for j = 1:test_num

        DOA=-L__:1:L__-1;
        DOA = DOA+rand(1,2*L__);
        DOA_ture(j,:) = DOA;
        for k=1:DOA_num
            [X,~]=signal_generate(M,snapshot,DOA(k),f0,fc,fs,1);
            X=awgn(X,SNR,'measured');
            [~,Rx]=feature_extract_R(X);
            temp=H'*reshape(Rx,M*M,1);
            temp=temp/norm(temp);
            S_est_d1(k,j,:,1)=real(temp);
            S_est_d1(k,j,:,2)=imag(temp);
        
            [X,~]=signal_generate(M,snapshot,DOA(k),f0,fc/alpha(i),fs,1);
            X=awgn(X,SNR,'measured');
            [~,Rx]=feature_extract_R(X);
            temp=H'*reshape(Rx,M*M,1);
            temp=temp/norm(temp);
            S_est_d2(k,j,:,1)=real(temp);
            S_est_d2(k,j,:,2)=imag(temp);

            
        end
    end
    disp(i);
    filename = fullfile('E:\code\array_tansfor\data\EX3',sprintf("EX3_Alpha_%s.mat", alpha_str(i)));
    save(filename,'DOA_ture','S_est_d1','S_est_d2')
end


