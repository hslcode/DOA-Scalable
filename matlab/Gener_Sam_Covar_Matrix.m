%Function that generates a sampling covariance matrix
%Author: Shulin Hu
%Parameter:
%       DOAs: The incident direction of the signals, unit: degree.
%       T   : The number of snapshot.
%       SNR_dB: The SNR of signal, unit: dB.
%       ULA_N: The number of array elements.
%       d:  The inter-element spaning relative to wavelength,
%       unit:lammda, where lammda is the carrier wavelength.
%Output:
%      Ry_sam: The  sampling covariance matrix with a shape of ULA_N*ULA_N. 
%      angles: The DoAs.
function [Ry_sam,angles] = Gener_Sam_Covar_Matrix(DOAs,T,SNR_dB,ULA_N,d)
    % The steering/response vector of the ULA
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
    angles(:) = DOAs';
end