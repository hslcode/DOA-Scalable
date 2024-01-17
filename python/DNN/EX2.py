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
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from ensemble_model import *
from utils import *
import tensorflow as tf
from sklearn.metrics import mean_squared_error
def asind(d):
    return np.degrees(np.arcsin(d))
def sind(d):
    rad = d/180*np.pi
    x = np.sin(rad)
    return x
# model path
model_path_nn = '../../model/DNN/initial_model_AI.npy'
model_path_sf = '../../model/DNN/spatialfilter_model_AI.npy'
model_path_ss = '../../model/DNN/spatialspectrum_model_AI.npy'


## base array (traning model) parameters
fc = 1e9     # carrier frequency
c = 3e8      # light speed
M = 16        # array sensor number
wavelength = c / fc  # signal wavelength
gamma_base = 0.5
d_base = gamma_base * wavelength  # inter-sensor distance

#Information about base array Geometry (for traning model) and bias array Geometry (by changing the inter-element spacing)
alpha = 0.5           # please replace the value For different geometries.
gamma_bias = gamma_base/alpha
d_bias = gamma_bias * wavelength
#save path
f_result_root = '../../result/EX2/'
f_result = f_result_root+'Result_EX2_Geometry_Generalization_Alpha_0.5.h5'
if not os.path.exists(f_result_root):
        os.makedirs(f_result_root)
# # spatial filter training parameters
theta_min = -60      # minimal DOA (degree)
theta_max = 60       # maximal DOA (degree)
grid_sf = 1         # DOA step (degree) for generating different scenarios
GRID_NUM_SF = int((theta_max - theta_min+1) / grid_sf)
SF_NUM = 6       # number of spatial filters

# # autoencoder parameters
input_size_sf = M * (M-1)
hidden_size_sf = int(1/2 * input_size_sf)
output_size_sf = input_size_sf
learning_rate_sf = 0.001

# # DNN parameters
grid_ss = 1    # inter-grid angle in spatial spectrum
NUM_GRID_SS = int((theta_max - theta_min + 0.5 * grid_ss) / grid_ss)   # spectrum grids
input_size_ss = M * (M-1)
hidden_size_ss = [int(2/3* input_size_ss), int(4/9* input_size_ss), int(1/3* input_size_ss)]
output_size_ss = int(NUM_GRID_SS / SF_NUM)
learning_rate_ss = 0.001

# array imperfection parameters
# We do not consider array imperfections
MC_mtx = np.identity(M)
AP_mtx = np.identity(M)
pos_para = np.zeros([M, 1])

# Load the DL model

tf.reset_default_graph()
enmod_3 = Ensemble_Model(input_size_sf=input_size_sf,
                         hidden_size_sf=hidden_size_sf,
                         output_size_sf=output_size_sf,
                         SF_NUM=SF_NUM,
                         learning_rate_sf=learning_rate_sf,
                         input_size_ss=input_size_ss,
                         hidden_size_ss=hidden_size_ss,
                         output_size_ss=output_size_ss,
                         learning_rate_ss=learning_rate_ss,
                         reconstruct_nn_flag=False,
                         train_sf_flag=False,
                         train_ss_flag=False,
                         model_path_nn=model_path_nn,
                         model_path_sf=model_path_sf,
                         model_path_ss=model_path_ss)
# # test data parameters
test_SNR = 20
T = 1000
theta_hat_max = np.floor(asind(alpha * sind(theta_max)))
test_DOA_set = np.arange(-theta_hat_max,theta_hat_max).astype(float)
GT_angles = test_DOA_set
DOA_num = len(test_DOA_set)

#Temporary variable to save results
DOA_base = []
DOA_bias = []
DOA_hat = []

# testing
with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print('testing...')
    for doa_index in range(DOA_num):    #Traverse the EFOV
        test_cov_vector = generate_array_cov_vector_AI(M,T, d_base, wavelength, test_DOA_set[doa_index], test_SNR, MC_mtx, AP_mtx, pos_para)
        data_batch = np.expand_dims(test_cov_vector, axis=-1)
        feed_dict = {enmod_3.data_train: data_batch}
        ss_output = sess.run(enmod_3.output_ss, feed_dict=feed_dict)
        ss_min = np.min(ss_output)
        ss_output_regularized = [ss if ss > -ss_min else [0.0] for ss in ss_output]
        doa_base = get_DOA_estimate(ss_output_regularized, test_DOA_set[doa_index], theta_min, grid_ss)          #The base estimation i.e. \theta_{base}
        DOA_base.append(doa_base)

        test_cov_vector = generate_array_cov_vector_AI(M,T, d_bias, wavelength, test_DOA_set[doa_index], test_SNR, MC_mtx, AP_mtx, pos_para)
        data_batch = np.expand_dims(test_cov_vector, axis=-1)
        feed_dict = {enmod_3.data_train: data_batch}
        ss_output = sess.run(enmod_3.output_ss, feed_dict=feed_dict)
        ss_min = np.min(ss_output)
        ss_output_regularized = [ss if ss > -ss_min else [0.0] for ss in ss_output]
        doa_bias = get_DOA_estimate(ss_output_regularized, asind(1/alpha*sind(test_DOA_set[doa_index])), theta_min, grid_ss)           #The bias estimation i.e. \theta_{bias}
        DOA_bias.append(doa_bias)

        doa_hat = asind(alpha * sind(doa_bias))     #The generalized estimation i.e. \hat_{\theta}
        DOA_hat.append(doa_hat)


# plot the ture DoA, generalization estimate, bias estimate and base estimate.
plt.figure()
plt.plot(GT_angles,GT_angles,'.',color='c',label='Ture DoA')
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
        if 'DOA_base_DNN' in hf:
            del hf['DOA_base_DNN']
        if 'DOA_bias_DNN' in hf:
                del hf['DOA_bias_DNN']
        if 'DOA_hat_DNN' in hf:
            del hf['DOA_hat_DNN']
        hf.create_dataset('DOA_ture', data=GT_angles)
        hf.create_dataset('DOA_base_DNN', data=DOA_base)
        hf.create_dataset('DOA_bias_DNN', data=DOA_bias)
        hf.create_dataset('DOA_hat_DNN', data=DOA_hat)
        hf.close()



