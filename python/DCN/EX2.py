'''
Experiment 2:  DoA Estimation Results for Varying Geometry
Creator: Shulin Hu
Date: 08/15/2023
run steps:
1. Generate Experiment 2 test data by running matlab/GENER_Test_Data_Ex2.m (except for DNN).
2. Select EX2.py under the corresponding method (DNN\DCN\CNN\CV-CNN).
3. Run the EX2.py.
4. The program will automatically save the ture DoAs, base estimates, bias estimates, and generalized estimates
to result/EX2/Result_EX2_Geometry_Generalization_Alpha_0.5.h5.
5. Run the matlab/Plot_Experiment_2.m to plot the results.
'''
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import h5py
import os
from sklearn.metrics import mean_squared_error
import random
def asind(d):
    return np.degrees(np.arcsin(d))
def sind(d):
    rad = d/180*np.pi
    x = np.sin(rad)
    return x
# Load the DL model
model_1DCNN = keras.models.load_model('../../model/1D_CNN/1DCNN_16.h5')
#Information about base array Geometry (for traning model) and bias array Geometry (by changing the inter-element spacing)
gamma_base = 0.5
alpha = 0.5
gamma_bias = gamma_base/alpha
# Load the test data.
f_data_root ='../../data/EX2/'
f_data = f_data_root+'EX2_Alpha_0.5.mat'
#save path
f_result_root = '../../result/EX2/'
f_result = f_result_root+'Result_EX2_Geometry_Generalization_Alpha_0.5.h5'
if not os.path.exists(f_result_root):
        os.makedirs(f_result_root)

#read test data
read_temp= scipy.io.loadmat(f_data)
S_est_gamma_base = read_temp['S_est_gamma_base']
S_est_gamma_bias = read_temp['S_est_gamma_bias']
GT_angles = read_temp['DOA_ture'].reshape(-1)
[DOA_num,grids,_] = np.shape(S_est_gamma_base)

#Temporary variable to save results
DOA_base = []
DOA_bias = []
DOA_hat = []


# testing
for doa_index in range(DOA_num):    #Traverse the EFOV
    DF_T_cnn = model_1DCNN.predict(S_est_gamma_base[doa_index, :, :].reshape(1, grids, 2))
    DF_T_cnn = np.array(DF_T_cnn)
    DF_T_cnn = np.reshape(DF_T_cnn, -1)
    doa_base = np.argmax(DF_T_cnn) - 60                 #The base estimation i.e. \theta_{base}
    DOA_base.append(doa_base)


    DF_T_cnn = model_1DCNN.predict(S_est_gamma_bias[doa_index, :, :].reshape(1, grids, 2))
    DF_T_cnn = np.array(DF_T_cnn)
    DF_T_cnn = np.reshape(DF_T_cnn, -1)
    doa_bias = np.argmax(DF_T_cnn) - 60                 #The bias estimation i.e. \theta_{bias}
    DOA_bias.append(doa_bias)

    doa_hat = asind(alpha * sind(doa_bias))              #The generalized estimation i.e. \hat_{\theta}
    DOA_hat.append(doa_hat)


# plot the ture DoA, generalization estimate, bias estimate and base estimate.
plt.figure()
plt.plot(GT_angles,DOA_base,'x',color='b',label='Base estimate')
plt.plot(GT_angles,DOA_bias,'*',color='g',label='Bias estimate')
plt.plot(GT_angles,DOA_hat,'+',color='r',label='Generalization estimate')
plt.xlabel("Direction (\u00b0)")
plt.ylabel("Estimated DoA (\u00b0)")
plt.legend()
plt.grid()
plt.show()

# Save the results.
with h5py.File(f_result, 'a') as hf:  
        if 'DOA_ture' in hf:
                del hf['DOA_ture']
        if 'DOA_base_1DCNN' in hf:
            del hf['DOA_base_1DCNN']
        if 'DOA_bias_1DCNN' in hf:
                del hf['DOA_bias_1DCNN']
        if 'DOA_hat_1DCNN' in hf:
            del hf['DOA_hat_1DCNN']
        hf.create_dataset('DOA_ture', data=GT_angles)
        hf.create_dataset('DOA_base_1DCNN', data=DOA_base)
        hf.create_dataset('DOA_bias_1DCNN', data=DOA_bias)
        hf.create_dataset('DOA_hat_1DCNN', data=DOA_hat)
        hf.close()

