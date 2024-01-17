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
import heapq
from sklearn.svm import SVR
from sklearn import preprocessing
# from sklearn.externals import joblib
import h5py
from sklearn.metrics import mean_squared_error
cnn_low = keras.models.load_model('cnn_16.h5')
#
f_data_root = '../../data/EX1/'
f_data = f_data_root+'EX3_Alpha_0.5.mat'

read_temp= scipy.io.loadmat(f_data)
K = 1
k = 1
d1 = 0.5
alpha = 0.5
d2 = d1/alpha

S_est_d2 = read_temp['S_est']
# R_est_d2 = read_temp['R_est']
DOA_ture = np.reshape(read_temp['DOA_ture'],-1)


# normalizer = preprocessing.Normalizer().fit(R_est_d2)
[r2, c] = np.shape(S_est_d2)
I = 120
DOA = np.arange(I) - 60


DCN = np.zeros((I, r2))

test_cnn = np.zeros((K, r2))
test_cnn_abs = np.zeros((K, r2))
est_DOA_d1 = []
est_DOA_d2 = []
est_phase_d1 = []
est_phase_d2 = []

for i in range(r2):

    DF_T_cnn = cnn_low.predict(S_est_d2[i, :, :].reshape(1, I, 2))
    DF_T_cnn = np.array(DF_T_cnn)
    DF_T_cnn = np.reshape(DF_T_cnn, -1)
    doa_d2 = np.argmax(DF_T_cnn)-60
    est_DOA_d2.append(doa_d2)
    phase_d2 = d1 * np.sin(doa_d2 / 180 * np.pi)
    est_phase_d2.append(phase_d2)

    doa_d1 = DOA_ture[i]
    est_DOA_d1.append(doa_d1)
    phase_d1 = d2 * np.sin(doa_d1 / 180 * np.pi)
    est_phase_d1.append(phase_d1)

pass
plt.figure()
plt.plot(DOA_ture,est_DOA_d1)
plt.plot(DOA_ture,est_DOA_d2)
plt.show()
plt.grid()

plt.figure()
plt.plot(DOA_ture,est_phase_d1)
plt.plot(DOA_ture,est_phase_d2)
plt.show()
plt.grid()

RMSE = np.sqrt(mean_squared_error(est_phase_d1, est_phase_d2))
print(RMSE)

# f_result_root = '../../../result/EX1/'
# f_result = f_result_root+'DCN_phase_alpha_0.5.h5'
# hf = h5py.File(f_result, 'w')
# hf.create_dataset('est_phase_d1', data=est_phase_d1)
# hf.create_dataset('est_phase_d2', data=est_phase_d2)
# hf.close()

