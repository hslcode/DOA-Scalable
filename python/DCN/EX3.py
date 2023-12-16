# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:50:00 2018

@author: me
"""
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
from sklearn import preprocessing
# from sklearn.externals import joblib
import h5py
from sklearn.metrics import mean_squared_error
def asind(d):
    return np.degrees(np.arcsin(d))
def sind(d):
    rad = d/180*np.pi
    x = np.sin(rad)
    return x
cnn_low = keras.models.load_model('cnn_16.h5')
#
f_data_root = '../../../data/EX3/'
f_data = f_data_root+'EX3_Alpha_0.8.mat'
read_temp= scipy.io.loadmat(f_data)
K = 1
k = 1
d1 = 0.5
alpha = 0.8
d2 = d1/alpha

S_est_d2 = read_temp['S_est_d2']


S_est_d1 = read_temp['S_est_d1']


DOA_ture = read_temp['DOA_ture']


[DOA_num,test_num,L,_] = np.shape(S_est_d2)
DOA = np.arange(L) - 60

est_DOA_d1 = []
est_DOA_d2 = []
est_phase_d1 = []
est_phase_d2 = []
RMSE_d1 = 0.0
RMSE_d2 = 0.0
for i in range(test_num):
    for j in range(DOA_num):
        DF_T_cnn = cnn_low.predict(S_est_d2[j,i, :, :].reshape(1, L, 2))
        DF_T_cnn = np.array(DF_T_cnn)
        DF_T_cnn = np.reshape(DF_T_cnn, -1)
        doa_d2 = np.argmax(DF_T_cnn)-60
        est_DOA_d2.append(asind(alpha*sind(doa_d2)))


        DF_T_cnn = cnn_low.predict(S_est_d1[j,i, :, :].reshape(1, L, 2))
        DF_T_cnn = np.array(DF_T_cnn)
        DF_T_cnn = np.reshape(DF_T_cnn, -1)
        doa_d1 = np.argmax(DF_T_cnn)-60
        est_DOA_d1.append(doa_d1)
    RMSE_d1 = RMSE_d1+np.sqrt(mean_squared_error(est_DOA_d1, DOA_ture[i,:]))
    RMSE_d2 = RMSE_d2+np.sqrt(mean_squared_error(est_DOA_d2, DOA_ture[i,:]))
    est_DOA_d1.clear()
    est_DOA_d2.clear()
print("RMSE_d1:",RMSE_d1/test_num)
print("RMSE_d2:",RMSE_d2/test_num)
# plt.figure()
# plt.plot(DOA_ture)
# plt.plot(est_DOA_d1)
# plt.grid()
# plt.show()
#
# plt.figure()
# plt.plot(DOA_ture)
# plt.plot(est_DOA_d2)
# plt.grid()
# plt.show()


# f_result_root = '../../../result/EX1/'
# f_result = f_result_root+'DCN_phase_alpha_0.5.h5'
# hf = h5py.File(f_result, 'w')
# hf.create_dataset('est_phase_d1', data=est_phase_d1)
# hf.create_dataset('est_phase_d2', data=est_phase_d2)
# hf.close()

